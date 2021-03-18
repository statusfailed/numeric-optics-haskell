# Numeric Optics

A Haskell implementation to (unofficially) accompany the paper
[Categorical Foundations of Gradient-Based Learning](https://arxiv.org/abs/2103.01931)
NOTE: this code is unfinished; it probably contains bugs.

The "official" code accompanying the paper can be found
[here](https://github.com/statusfailed/numeric-optics-python).

# Warning: Unfinished

This code is essentially incomplete, but I'm publishing it anyway because some
of the aspects of the paper are a bit clearer in Haskell.

You should be able to build with 

    cabal build

But note you'll need an installation of CUDA for this to work (see the
"Building" section of the README).

You can run the demo program with

    cabal run

which runs 100 iterations of SGD on the Iris dataset (but doesn't do much with
the results!)

# Code

Since I've uploaded just in case someone finds it useful, here are some pointers
to relevant modules, the source of which can be found in the [src](./src)
directory.

- `Control.Categories`
  - An alternate typeclass for Categories, with composition
    `(~>) :: cat a b -> cat b c -> cat a c`
  - Typeclasses for Cartesian and Monoidal categories
  - Some shorthand/infix operators, e.g., `π0` and `π1` for projections, and infix
    tensor product `×`
- `Numeric.Optics.Types`
  - the `MonoLens` type (monomorphic lenses)
  - Monoidal instance for `MonoLens`
  - the `Para` type, and parametrised composition `(~~>)`
- `Numeric.Optics.Base.Accelerate`
  - Category instances for `DSL Acc` - a type wrapping the [Accelerate DSL][accelerate]
- `Numeric.Optics.Base.Accelerate.NeuralNetwork`
  - Specific morphisms for building models, e.g.: `dense`, a single dense layer
    (as in the paper)

# Building

This project uses the [Accelerate][accelerate] library for GPU acceleration.
Note that you will therefore need several system dependencies (e.g., cuda for
the GPU backend) in order to build.

[accelerate]: https://hackage.haskell.org/package/accelerate
