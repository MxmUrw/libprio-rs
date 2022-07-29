// SPDX-License-Identifier: MPL-2.0

//! Some functions shared by several [`Type`](crate::flp::Type) implementations.

use crate::field::FieldElement;
use crate::flp::{FlpError, Gadget, Type};
use std::convert::TryFrom;

/// Interpret `bits` as `F::Integer` if it's representable in that type and smaller than the field modulus.
pub(crate) fn valid_int_from_usize<F: FieldElement>(bits: usize) -> Result<F::Integer, FlpError> {
    let bits_int = F::Integer::try_from(bits).map_err(|err| {
        FlpError::Encode(format!(
            "bit length ({}) cannot be represented as a field element: {:?}",
            bits, err,
        ))
    })?;

    if F::modulus() >> bits_int == F::Integer::from(F::zero()) {
        return Err(FlpError::Encode(format!(
            "bit length ({}) exceeds field modulus",
            bits,
        )));
    }
    Ok(bits_int)
}

/// Encode `input` as bitvector of elements of `T::Field`. The number of bits is given
/// by the length of the `output` slice.
///
/// # Arguments
///
/// * `input` - The field element to encode
/// * `max_input` - The maximal number representable with `output.len()` bits. Should be equal to `output.len()^2-1`
/// * `output` - The slice to write the encoded bits into. Least signicant bit comes first
pub(crate) fn encode_into_bitvector_representation_slice<F: FieldElement>(
    input: &F::Integer,
    max_input: F::Integer,
    output: &mut [F],
) -> Result<(), FlpError> {
    if *input > max_input {
        return Err(FlpError::Encode(
            "value of input exceeds bit length".to_string(),
        ));
    }

    let one = F::Integer::from(F::one());
    for (l, bit) in output.iter_mut().enumerate() {
        let l = F::Integer::try_from(l).unwrap();
        let w = F::from((*input >> l) & one);
        *bit = w;
    }
    Ok(())
}

/// Encode `input` as `bits`-bit vector of elements of `T::Field` if it's small enough
/// to be represented with that many bits.
///
/// # Arguments
///
/// * `input` - The field element to encode
/// * `bits` - The number of bits to use for the encoding
/// * `max_input` - The maximal number representable with `bits` bits. Should be equal to `bits^2-1`
pub(crate) fn encode_into_bitvector_representation<F: FieldElement>(
    input: &F::Integer,
    bits: usize,
    max_input: F::Integer,
) -> Result<Vec<F>, FlpError> {
    let mut result = vec![F::zero(); bits];
    encode_into_bitvector_representation_slice(input, max_input, &mut result)?;
    Ok(result)
}

/// Decode the bitvector-represented value `input` into a simple representation as a single field element.
pub(crate) fn decode_from_bitvector_representation<F: FieldElement>(input: &[F]) -> F {
    let mut decoded = F::zero();
    for (l, bit) in input.iter().enumerate() {
        let w = F::from(F::Integer::try_from(1 << l).unwrap());
        decoded += w * *bit;
    }
    decoded
}

/// Compute a random linear combination of the result of calls of `g` on each element of `input`.
///
/// # Arguments
///
/// * `g` - The gadget to be applied elementwise
/// * `input` - The vector on whose elements to apply `g`
/// * `rnd` - The randomness used for the linear combination
pub(crate) fn call_gadget_on_vec_entries<F: FieldElement>(
    g: &mut Box<dyn Gadget<F>>,
    input: &[F],
    rnd: F,
) -> Result<F, FlpError> {
    let mut range_check = F::zero();
    let mut r = rnd;
    for chunk in input.chunks(1) {
        range_check += r * g.call(chunk)?;
        r *= rnd;
    }
    Ok(range_check)
}

/// Check whether `input` and `joint_rand` have the length expected by `typ`.
pub(crate) fn valid_call_check<T: Type>(
    typ: &T,
    input: &[T::Field],
    joint_rand: &[T::Field],
) -> Result<(), FlpError> {
    if input.len() != typ.input_len() {
        return Err(FlpError::Valid(format!(
            "unexpected input length: got {}; want {}",
            input.len(),
            typ.input_len(),
        )));
    }

    if joint_rand.len() != typ.joint_rand_len() {
        return Err(FlpError::Valid(format!(
            "unexpected joint randomness length: got {}; want {}",
            joint_rand.len(),
            typ.joint_rand_len()
        )));
    }

    Ok(())
}

/// Check if the length of `input` matches `typ`'s `input_len()`.
pub(crate) fn truncate_call_check<T: Type>(typ: &T, input: &[T::Field]) -> Result<(), FlpError> {
    if input.len() != typ.input_len() {
        return Err(FlpError::Truncate(format!(
            "Unexpected input length: got {}; want {}",
            input.len(),
            typ.input_len()
        )));
    }

    Ok(())
}

/// Given a vector `data` of field elements which should contain exactly one entry, return the integer representation of that entry.
pub(crate) fn decode_result<F: FieldElement>(data: &[F]) -> Result<F::Integer, FlpError> {
    if data.len() != 1 {
        return Err(FlpError::Decode("unexpected input length".into()));
    }
    Ok(F::Integer::from(data[0]))
}

/// Given a vector `data` of field elements, return a vector containing the corresponding integer representations, if the number of entries matches `expected_len`.
pub(crate) fn decode_result_vec<F: FieldElement>(
    data: &[F],
    expected_len: usize,
) -> Result<Vec<F::Integer>, FlpError> {
    if data.len() != expected_len {
        return Err(FlpError::Decode("unexpected input length".into()));
    }
    Ok(data.iter().map(|elem| F::Integer::from(*elem)).collect())
}
