// SPDX-License-Identifier: MPL-2.0

//! Implementations of encoding fixed point types as field elements and field elements as floats for the [`FixedPointL2BoundedVecSum`](crate::flp::fixedpoint_l2::FixedPointL2BoundedVecSum) type.

use crate::field::{Field64, FieldElement};
use crate::flp::types::fixedpoint_l2::CompatibleFloat;
use fixed::FixedI16;

impl<U> CompatibleFloat<Field64> for FixedI16<U> {
    type Float = f64;
    fn to_float(d: Field64, c: usize) -> f64 {
        // get integer representation of field element
        let i: u64 = <Field64 as FieldElement>::Integer::from(d);

        // get field modulus
        let p: u64 = Field64::modulus();

        // elements in the left half of the field are positive,
        // in the right half are negative
        let half = p / 2;

        // check whether the given integer was originally negative or positive
        //  - positive: `i \in [0 , 2^(n-1)-1]`
        //  - negative: `i \in [p-2^(n-1) , p-1]`
        if i < half {
            let f = i as f64;
            f * f64::powi(2.0, -15)
        } else {
            let f = (p - i) as f64;
            -f * f64::powi(2.0, -15)
        }
    }

    fn to_field_integer(fp: FixedI16<U>) -> <Field64 as FieldElement>::Integer {
        if fp.is_negative() {
            let modulus = Field64::modulus();
            let i = fp.abs().to_bits() as u64;
            return modulus - i;
        } else {
            return fp.to_bits() as u64;
        }
    }
}
