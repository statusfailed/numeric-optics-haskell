{-# LANGUAGE PolyKinds       #-}
{-# LANGUAGE KindSignatures  #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeFamilies    #-}
-- ^ Needed for 'Cat'
{-# LANGUAGE NoStarIsType    #-}
-- ^ Use Type instead of * in kind signatures
{-# LANGUAGE AllowAmbiguousTypes #-}
-- ^ Needed for Monoidal class methods (not tensor though- weird!)
-- Experimental
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
module Control.Categories where

import qualified Prelude
import Prelude hiding ((.), id)

import GHC.Exts (Constraint)
import GHC.Types (Type)

-- | The class of types of /morphism/ of a category.
-- A simple example is to take cat as '(->)', the type of functions,
-- and objects as any Haskell 'Type'.
-- This class is similar to 'Control.Category.Category', but where the kind of
-- objects can be arbitrarily constrained using the associated 'Obj' type
-- family.
-- NOTE: composition with (~>) is in /diagrammatic order/ (left-to-right).
class Cat (cat :: obj -> obj -> Type) where
  type Obj cat (a :: obj) :: Constraint
  
  id :: forall a . Obj cat a => cat a a
  (~>) :: forall a b c . (Obj cat a, Obj cat b, Obj cat c) => cat a b -> cat b c -> cat a c

-- | Monoidal categories
class Cat cat => Monoidal (cat :: obj -> obj -> Type)  where
  type Tensor cat (a :: obj) (b :: obj) :: obj
  type Unit   cat                       :: obj

  tensor :: forall a b c d . (Obj cat a, Obj cat b, Obj cat c, Obj cat d)
         => cat a b
         -> cat c d
         -> cat (Tensor cat a c) (Tensor cat b d)

  assocL :: forall a b c . (Obj cat a, Obj cat b, Obj cat c, Obj cat (Tensor cat a b), Obj cat (Tensor cat b c))
         => cat (Tensor cat (Tensor cat a b) c) (Tensor cat a (Tensor cat b c))

  assocR :: forall a b c . (Obj cat a, Obj cat b, Obj cat c, Obj cat (Tensor cat a b), Obj cat (Tensor cat b c))
         => cat (Tensor cat a (Tensor cat b c)) (Tensor cat (Tensor cat a b) c)

  unitorL   :: forall a . (Obj cat a, Obj cat (Tensor cat (Unit cat) a)) => cat (Tensor cat (Unit cat) a) a
  counitorL :: forall a . (Obj cat a, Obj cat (Tensor cat (Unit cat) a)) => cat a (Tensor cat (Unit cat) a)

-- | Cartesian categories
class Monoidal cat => Cartesian cat where
  proj0 :: forall a b . (Obj cat a, Obj cat b, Obj cat (Tensor cat a b))
        => cat (Tensor cat a b) a

  proj1 :: forall a b . (Obj cat a, Obj cat b, Obj cat (Tensor cat a b))
        => cat (Tensor cat a b) b

  pair  :: forall q a b . (Obj cat q, Obj cat a, Obj cat b, Obj cat (Tensor cat a b))
        => cat q a -> cat q b -> cat q (Tensor cat a b)

-------------------------------
-- Cartesian category morphisms

-- Infix operator for `tensor`
(×) :: forall cat a b c d .
  ( Monoidal cat
  , Obj cat a, Obj cat b, Obj cat c, Obj cat d
  , Obj cat (Tensor cat a c), Obj cat (Tensor cat b d)
  )
  => cat a b
  -> cat c d
  -> cat (Tensor cat a c) (Tensor cat b d)
(×) = tensor

paired :: forall cat q a b . 
  (Cartesian cat, Obj cat q, Obj cat a, Obj cat b, Obj cat (Tensor cat a b))
  => cat q a -> cat q b -> cat q (Tensor cat a b)
paired = pair

-- | Unicode shorthand for proj0
π0 :: forall (cat :: obj -> obj -> Type) (a :: obj) (b :: obj) .
  (Cartesian cat, Obj cat a, Obj cat b, Obj cat (Tensor cat a b))
  => cat (Tensor cat a b) a
π0 = proj0 @cat @a @b

-- | Unicode shorthand for proj1
π1 :: forall (cat :: obj -> obj -> Type) (a :: obj) (b :: obj) .
  (Cartesian cat, Obj cat a, Obj cat b, Obj cat (Tensor cat a b))
  => cat (Tensor cat a b) b
π1 = proj1 @cat @a @b

-- | The twist (σ) morphism in a cartesian category, i.e. @<π1, π0>@
twist :: forall cat a b .
  ( Cartesian cat
  , Obj cat a, Obj cat b     -- A, B \in C
  , Obj cat (Tensor cat a b) -- A × B is an object
  , Obj cat (Tensor cat b a) -- B × A is an object
  , Obj cat (Tensor cat (Tensor cat a b) (Tensor cat a b)) -- (A × B) × (A × B) \in C
  )
  => cat (Tensor cat a b) (Tensor cat b a)
twist = pair @cat @(Tensor cat a b) (proj1 @cat @a @b) (proj0 @cat @a @b)

-------------------------------
-- Common instances

instance Cat (->) where
  type Obj (->) a = ()
  id = Prelude.id
  (~>) = flip (Prelude..)

instance Monoidal (->) where
  type Tensor (->) a b = (a, b)
  type Unit   (->)     = ()

  tensor f g (a, c) = (f a, g c)
  
  unitorL (_, a) = a
  counitorL a = ((), a)

  assocL ((a, b), c) = (a, (b, c))
  assocR (a, (b, c)) = ((a, b), c)

instance Cartesian (->) where
  proj0 = fst
  proj1 = snd
  pair f g q = (f q, g q)
