# zk_dtypes Testing Guide

## Running Tests
```
bazel test //...
```

## Property-Based Testing
New field types MUST verify:
- Addition: associativity, commutativity, identity (0), inverse
- Multiplication: associativity, commutativity, identity (1), inverse (non-zero)
- Distributivity: a * (b + c) == a*b + a*c
- Montgomery conversion: from_mont(to_mont(x)) == x

## Test Naming
- Test files: `*_test.py`
- Test functions: `test_<field_type>_<property>`
