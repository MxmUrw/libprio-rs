// SPDX-License-Identifier: MPL-2.0

//! A [`Type`](crate::flp::Type) for summing vectors of fixed point numbers whose L2 norm is bounded by 1.

use crate::field::FieldElement;
use crate::flp::gadgets::PolyEval;
use crate::flp::types_utilities::{
    decode_from_bitvector_representation, encode_into_bitvector_representation_slice,
    truncate_call_check, valid_call_check,
};
use crate::flp::{FlpError, Gadget, Type};
use crate::polynomial::poly_range_check;
use fixed::traits::Fixed;

use std::{
    convert::{TryFrom, TryInto},
    fmt::Debug,
    marker::PhantomData,
};

pub mod compatible_float;

/// Assign a `Float` type to this type and describe how to represent this type as an integer of the given
/// field, and how to represent a field element as the assigned `Float` type.
pub trait CompatibleFloat<F: FieldElement> {
    /// The float type F can be converted into.
    type Float: Debug + Clone;
    /// Represent a field element as `Float`, given the number of clients `c`.
    fn to_float(t: F, c: usize) -> Self::Float;
    /// Represent a value of this type as an integer in the given field.
    fn to_field_integer(t: Self) -> <F as FieldElement>::Integer;
}

/// The fixed point vector sum data type. Each measurement is a vector of fixed point numbers of type T, and the
/// aggregate result is the sum of the measurements.
///
/// The validity circuit verifies that the L2 norm of each measurment is bounded by 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedPointL2BoundedVecSum<T: Fixed, F: FieldElement> {
    bits_per_entry: usize,
    entries: usize,
    bits_for_norm: usize,
    one: <F as FieldElement>::Integer,
    max_entry: <F as FieldElement>::Integer,
    max_encoded_norm: <F as FieldElement>::Integer,
    range_01_checker: Vec<F>,
    square_computer: Vec<F>,
    phantom: PhantomData<T>,
    // range/position constants
    range_norm_begin: usize,
    range_norm_end: usize,
}

// layout:
//
// |---- bits_per_entry * entries ----|---- bits_for_norm ----|
//   ^                                  ^
//   \- the input vector entries        |
//                                      \- the encoded norm
//

impl<T: Fixed, F: FieldElement> FixedPointL2BoundedVecSum<T, F> {
    /// Return a new [`FixedPointL2BoundedVecSum`] type parameter. Each value of this type is a fixed point vector with
    /// `entries` entries.
    pub fn new(entries: usize) -> Result<Self, FlpError> {
        if <T as Fixed>::INT_NBITS != 1 {
            return Err(FlpError::Encode(format!(
                "Expected fixed point type with one integer bit, but got {}.",
                <T as Fixed>::INT_NBITS,
            )));
        }

        ///////////////////////////
        // number of bits of an entry
        let bits_per_entry = (<T as Fixed>::INT_NBITS + <T as Fixed>::FRAC_NBITS)
            .try_into()
            .unwrap();

        ///////////////////////////
        // The norm of our vector should be less than `2^(2*(bits - 1))`
        // This means that a valid norm is given exactly by a binary
        // number with the following number of bits.
        let bits_for_norm = 2 * (bits_per_entry - 1);

        let bits_per_entry_fei =
            <F as FieldElement>::Integer::try_from(bits_per_entry).map_err(|err| {
                FlpError::Encode(format!(
                    "entry bit length ({}) cannot be represented as a field element: {:?}",
                    bits_per_entry, err,
                ))
            })?;

        let fzero = <F as FieldElement>::Integer::from(<F as FieldElement>::zero());
        if <F as FieldElement>::modulus() >> bits_per_entry_fei == fzero {
            return Err(FlpError::Encode(format!(
                "bit length ({}) exceeds field modulus",
                bits_per_entry,
            )));
        }

        let bits_for_norm_fei =
            <F as FieldElement>::Integer::try_from(bits_for_norm).map_err(|err| {
                FlpError::Encode(format!(
                    "norm bit length ({}) cannot be represented as a field element: {:?}",
                    bits_for_norm, err,
                ))
            })?;

        let fone = <F as FieldElement>::one();
        let one = <F as FieldElement>::Integer::from(fone);

        let max_entry = (one << bits_per_entry_fei) - one;
        let max_encoded_norm = (one << bits_for_norm_fei) - one;

        ///////////////////////////
        // make sure that the maximal value that the norm can take fits into our field
        // it is: `entries * 2^(2*bits + 1)`
        println!("bits is {bits_per_entry}");
        let usize_max_norm_value: usize = entries * (1 << (2 * bits_per_entry + 1));
        let max_norm_value =
            <F as FieldElement>::Integer::try_from(usize_max_norm_value).map_err(|err| {
                FlpError::Encode(format!(
                    "bit length ({}) cannot be represented as a FieldElement::Integer: {:?}",
                    bits_per_entry, err,
                ))
            })?;

        if max_norm_value > <F as FieldElement>::modulus() {
            return Err(FlpError::Encode(format!(
                "The maximal norm value ({}) exceeds field modulus",
                usize_max_norm_value,
            )));
        }
        println!(
            "The max norm_value is: {:?}\nThe field modulus is: {:?}",
            max_norm_value,
            <F as FieldElement>::modulus()
        );

        ///////////////////////////
        // return the constructed self
        let res = Ok(Self {
            bits_per_entry,
            entries,
            bits_for_norm,
            one,
            max_entry,
            max_encoded_norm,
            range_01_checker: poly_range_check(0, 2),
            square_computer: vec![F::zero(), F::zero(), F::one()],
            phantom: PhantomData,

            // range constants
            range_norm_begin: entries * bits_per_entry,
            range_norm_end: entries * bits_per_entry + bits_for_norm,
        });
        res
    }
}

/////////////////////////////////////////////////////////////////////////////////
// helper functions

// computing the square of the L2 norm of a vector of field elements.
// the entries are encoded fixed-point numbers in [-1,1) represented as field elements in [0, 2^n)
// where n is the number of bits the fixed-point representation had. ie to encode fixed-point x:
// enc(x) = 2^(n-1) * x + 2^(n-1)
// with inverse
// dec(y) = (y - 2^(n-1)) * 2^(1-n)
// to compute the sum of the squares of fixed-point numbers, we need to compute the following sum
// on the field element encoding:
// sum for y in entries: dec(y)^2 = (y^2 - y*2^n + 2^(2n-2)) * 2^(2-2n)
// we omit computation of the latter factor, as it is constant in the input and we only want to
// compare with the claimed norm.
// as the constant summand 2^(2n-2) is distributed among the clients, we multiply with a share of 1
fn compute_norm_of_entries<F, Fs, SquareFun, E>(
    entries: Fs,
    bits_per_entry: usize,
    constant_part_multiplier: F,
    sq: &mut SquareFun,
) -> Result<F, E>
where
    F: FieldElement,
    Fs: IntoIterator<Item = F>,
    SquareFun: FnMut(F) -> Result<F, E>,
{
    //--------------------------------------------
    // norm computation
    //
    // We need to ensure that norm(entries) = claimed_norm
    // let entries = &input[0..self.entries*self.bits_per_entry];
    let mut computed_norm = F::zero();
    //
    // constants
    let constant_part = F::Integer::try_from(1 << (2 * bits_per_entry - 2)).unwrap(); // = 2^(2n-2)
    let linear_part = F::Integer::try_from(1 << (bits_per_entry)).unwrap(); // = 2^n

    for entry in entries.into_iter() {
        let summand = sq(entry)? + F::from(constant_part) * constant_part_multiplier
            - F::from(linear_part) * (entry);
        computed_norm += summand;
    }
    Ok(computed_norm)
}

impl<T: Fixed, F: FieldElement> Type for FixedPointL2BoundedVecSum<T, F>
where
    T: CompatibleFloat<F>,
{
    type Measurement = Vec<T>;
    type AggregateResult = Vec<<T as CompatibleFloat<F>>::Float>;
    type Field = F;

    fn encode_measurement(&self, fp_summands: &Vec<T>) -> Result<Vec<Self::Field>, FlpError> {
        // first convert all my entries to the field-integers
        let mut integer_entries: Vec<<Self::Field as FieldElement>::Integer> =
            Vec::with_capacity(self.entries);
        for fp_summand in fp_summands {
            let summand = &<T as CompatibleFloat<Self::Field>>::to_field_integer(*fp_summand);
            if *summand > self.max_entry {
                return Err(FlpError::Encode(
                    "value of summand exceeds bit length".to_string(),
                ));
            }
            integer_entries.push(*summand);
        }

        //-------------------------------------------------------
        // vector entries
        //-------------------------------------------------------
        //
        // then encode them bitwise
        let mut encoded: Vec<Self::Field> =
            vec![Self::Field::zero(); self.bits_per_entry * self.entries + self.bits_for_norm];
        //
        for (l, entry) in integer_entries.clone().iter().enumerate() {
            encode_into_bitvector_representation_slice(
                entry,
                self.max_entry,
                &mut encoded[l * self.bits_per_entry..(l + 1) * self.bits_per_entry],
            )?;
        }

        //-------------------------------------------------------
        // norm
        //-------------------------------------------------------
        //
        // compute the norm
        let field_entries = integer_entries.iter().map(|&x| Self::Field::from(x));
        let norm = compute_norm_of_entries::<_, _, _, FlpError>(
            field_entries,
            self.bits_per_entry,
            Self::Field::from(self.one),
            &mut |x| Ok(x * x),
        )?;
        let norm_int = <Self::Field as FieldElement>::Integer::from(norm);
        println!("Computed norm of entries: {norm_int:?}");
        //
        println!("The bit encoding is: ");
        //
        // push the bits of the norm
        encode_into_bitvector_representation_slice(
            &norm_int,
            self.max_encoded_norm,
            &mut encoded[self.range_norm_begin..self.range_norm_end],
        )?;
        println!("");

        // return
        Ok(encoded)
    }

    fn decode_result(
        &self,
        data: &[Self::Field],
        num_measurements: usize,
    ) -> Result<Vec<<T as CompatibleFloat<Self::Field>>::Float>, FlpError> {
        if data.len() != self.entries {
            return Err(FlpError::Decode("unexpected input length".into()));
        }
        let mut res = vec![];
        for d in data {
            // to decode a single integer, we'd use the function
            // dec(y) = (y - 2^(n-1)) * 2^(1-n) = y * 2^(1-n) - 1
            // as d is the sum of c encoded vector entries where c is the number of clients, we
            // compute d * 2^(1-n) - c
            let decoded = <T as CompatibleFloat<Self::Field>>::to_float(*d, num_measurements);
            res.push(decoded);
        }
        Ok(res)
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<Self::Field>>> {
        println!("Inside gadget! ========================");
        // We need two gadgets:
        //
        // (0): check that field element is 0 or 1
        let gadget0 = PolyEval::new(
            self.range_01_checker.clone(),
            self.bits_per_entry * self.entries + self.bits_for_norm,
        );
        //
        // (1): compute square of field element
        let gadget1 = PolyEval::new(self.square_computer.clone(), self.entries);

        let res: Vec<Box<dyn Gadget<Self::Field>>> = vec![Box::new(gadget0), Box::new(gadget1)];
        println!("Gadget ended ! ========================");
        res
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<Self::Field>>>,
        input: &[Self::Field],
        joint_rand: &[Self::Field],
        _num_shares: usize,
    ) -> Result<Self::Field, FlpError> {
        println!("Inside valid! ====================");
        valid_call_check(self, input, joint_rand)?;

        //--------------------------------------------
        // range checking
        //
        // (I) for encoded input vector entries
        //    We need to make sure that all the input vector entries
        //    contain only 0/1 field elements.
        //
        // (II) for encoded norm
        //    The norm should also be encoded by 0/1 field elements.
        //    Every such encoded number represents a valid norm.
        //
        // Since all input vector entry (field-)bits, as well as the norm bits, are contiguous,
        // we do the check directly for all bits [0..entries*bits_per_entry + bits_for_norm].
        //
        // Check that each element is a 0 or 1:
        let mut validity_check = Self::Field::zero();
        let mut r = joint_rand[0];
        for chunk in input[0..self.range_norm_end].chunks(1) {
            validity_check += r * g[0].call(chunk)?;
            r *= joint_rand[0];
        }

        //--------------------------------------------
        // norm computation
        //
        // an iterator over the decoded entries
        println!("before entry decoding");
        let decoded_entries = input[0..self.entries * self.bits_per_entry]
            .chunks(self.bits_per_entry)
            .map(decode_from_bitvector_representation);
        //
        // the constant bit
        let num_of_clients = <Self::Field as FieldElement>::Integer::try_from(_num_shares).unwrap();
        let constant_part_multiplier =
            Self::Field::from(self.one) / Self::Field::from(num_of_clients);
        //
        // the computed norm
        let computed_norm = compute_norm_of_entries(
            decoded_entries,
            self.bits_per_entry,
            constant_part_multiplier,
            &mut |x| g[1].call(std::slice::from_ref(&x)),
        )?;
        //
        // the claimed norm
        let claimed_norm_enc = &input[self.range_norm_begin..self.range_norm_end];
        println!("before norm decoding, it is: {:?}", claimed_norm_enc);
        println!("parameters are: \nself.entries: {}\nself.bits_per_entry: {}\nself.bits_for_norm: {}\nnorm_length: {}", self.entries, self.bits_per_entry, self.bits_for_norm, claimed_norm_enc.len());
        let claimed_norm = decode_from_bitvector_representation(claimed_norm_enc);
        //
        // add the check that computed norm == claimed norm
        validity_check += r * (computed_norm - claimed_norm);
        r *= joint_rand[0];
        println!("////////////////////////////////////////");
        println!("Computed norm: {computed_norm:?}");
        println!("Claimed  norm: {claimed_norm:?}");
        println!("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\");

        // Return the result
        println!("Valid ended! ====================");
        Ok(validity_check)
    }

    fn truncate(&self, input: Vec<Self::Field>) -> Result<Vec<Self::Field>, FlpError> {
        println!("Inside truncate!! ===========================");
        println!("input: {:?}", input);
        truncate_call_check(self, &input)?;

        let mut decoded_vector = vec![];

        for i_entry in 0..self.entries {
            let start = i_entry * self.bits_per_entry;
            let end = (i_entry + 1) * self.bits_per_entry;

            let mut decoded = Self::Field::zero();
            for (l, bit) in input[start..end].iter().enumerate() {
                let w = Self::Field::from(
                    <Self::Field as FieldElement>::Integer::try_from(1 << l).unwrap(),
                );
                decoded += w * *bit;
            }
            decoded_vector.push(decoded);
        }
        Ok(decoded_vector)
    }

    fn input_len(&self) -> usize {
        self.bits_per_entry * self.entries + self.bits_for_norm
    }

    fn proof_len(&self) -> usize {
        // computed via
        // `gadget.arity() + gadget.degree() * ((1 + gadget.calls()).next_power_of_two() - 1) + 1;`
        //
        let proof_gadget_0 = 2
            * ((1 + (self.bits_per_entry * self.entries + self.bits_for_norm)).next_power_of_two()
                - 1)
            + 2;
        let proof_gadget_1 = 2 * ((1 + self.entries).next_power_of_two() - 1) + 2;
        println!("we have lengths:\n{proof_gadget_0}\n{proof_gadget_1}");
        proof_gadget_0 + proof_gadget_1
    }

    fn verifier_len(&self) -> usize {
        5 // why?
    }

    fn output_len(&self) -> usize {
        self.entries
    }

    fn joint_rand_len(&self) -> usize {
        1
    }

    fn prove_rand_len(&self) -> usize {
        2
    }

    fn query_rand_len(&self) -> usize {
        2
    }
}
