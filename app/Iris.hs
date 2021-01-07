{-# LANGUAGE DeriveGeneric      #-}
{-# LANGUAGE DeriveAnyClass     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE FlexibleContexts #-}
module Iris where

import Prelude hiding (id, Ord, Num, (<), (==))
import qualified Prelude as P

import GHC.Generics
import Data.Foldable (toList)
import Control.Monad

-- Loading data
import Data.Csv
import Data.ByteString.Lazy (ByteString)
import qualified Data.ByteString.Lazy as BL

-- Accelerate
import Data.Array.Accelerate (Exp, Acc, Matrix, Vector, Z(..), (:.)(..), pattern T2, pattern I1, fill, constant, (!), (?), (==), (<))
import qualified Data.Array.Accelerate as A

-- Modelling
import Numeric.Optics
import Numeric.Optics.Base.Accelerate
import Numeric.Optics.Base.Accelerate.NeuralNetwork as NN

-- random numbers for parameter initialization
import qualified Data.Array.Accelerate.System.Random.MWC as Random
import qualified System.Random.MWC.Distributions         as MWC

import Training

data IrisClass = Setosa | Versicolor | Virginica
  deriving(Eq, P.Ord, Read, Show, Enum, Bounded, Generic, A.Elt)

instance FromField IrisClass where
  parseField s
    | s P.== "Iris-setosa" = pure Setosa
    | s P.== "Iris-versicolor" = pure Versicolor
    | s P.== "Iris-virginica" = pure Virginica
    | otherwise = mzero

data Iris = Iris
  { sepal_length :: Double
  , sepal_width :: Double
  , petal_length :: Double
  , petal_width :: Double
  , iris_class  :: IrisClass
  } deriving(Eq, P.Ord, Read, Show, Generic)

instance FromRecord Iris

irisData :: IO [Iris]
irisData = either error toList . decode HasHeader <$> BL.readFile "data/iris.csv"

toAccelerateData :: [Iris] -> Matrix Double
toAccelerateData xs = A.fromList (Z :. n :. 4) (xs >>= f)
  where
    f (Iris a b c d _) = [a,b,c,d]
    n = length xs

-- NOTE: we encode labels as Ints, since working with sum types in Accelerate
-- is a bit painful.
toAccelerateLabels :: [Iris] -> Matrix Int
toAccelerateLabels xs = A.fromList (Z :. n :. 1) $ fmap (fromEnum . iris_class) xs
  where n = length xs

-- One-hot version of toAccelerateLabels
toAccelerateLabelsOneHot :: [Iris] -> Matrix Int
toAccelerateLabelsOneHot xs = A.fromList (Z :. n :. 3) $ xs >>= oneHot
  where
    n = length xs
    oneHot i = case iris_class i of
                 Setosa     -> [1, 0, 0]
                 Versicolor -> [0, 1, 0]
                 Virginica  -> [0, 0, 1]

-------------------------------
-- Models

-- A very simple model with no parameters that simply projects out the final
-- dimension of the data and clips it to the range [0, 2]
-- NOTE: this model works quite well only because I've looked at the data and
-- constructed it so!
baselineModel :: Acc (Vector Double) -> Acc (Vector Int)
baselineModel = toVector . A.unit . f . (! (I1 3))
  where
    f = clip 0 2 . A.floor

-- Same as the @baselineModel@, but with one-hot-encoded labels
baselineModelOneHot :: Acc (Vector Double) -> Acc (Vector Int)
baselineModelOneHot v = f (v ! (I1 3))
  where f x = A.generate (I1 3) (\(I1 i) -> i == clip 0 2 (A.floor x) ? (1, 0))

-- Same as the @baselineModel@, but expressed using the Numeric Optics library
baselineModelLens :: ParaLens (,) (DSL Acc) () (Vector Double) (Vector Int)
baselineModelLens = unitorL ~> projection (I1 3) ~> reshapeLens (I1 1) ~> mapLens efloor

-------------------------------
-- Baseline model, but where input data is first scaled pointwise.

scaleModelParams :: Acc (Vector Double, Vector Double)
scaleModelParams = T2 (A.fill (I1 4) 0.0) (A.fill (I1 4) 0.0) 

scaleModel :: ParaLens (,) (DSL Acc) (Vector Double, Vector Double) (Vector Double) (Vector Int)
scaleModel = zipWithLens multiply ~~> zipWithLens add ~> projection (I1 3) ~> reshapeLens (I1 1) ~> mapLens efloor

-------------------------------
-- Neural network models

-- Parameters for a single dense layer
type DenseParams = (Vector Double, Matrix Double)

-------------------------------
-- same as @scaleModel@ model, but using NN.dense instead of explicit scaling
-- Note that this model gets the same accuracy as @scaleModel@
scaleDense :: ParaLens (,) (DSL Acc) DenseParams (Vector Double) (Vector Int)
scaleDense = tensor id (projection (I1 3) ~> reshapeLens (I1 1)) ~> NN.dense id ~> mapLens efloor

-- Initialize dense layer parameters using mean @0@ and stddev @0.01@
-- We follow the defaults for the RandomNormal initializer of
-- [Keras](https://keras.io/api/layers/initializers/)
mkDenseParams :: Int -> Int -> IO DenseParams
mkDenseParams a b = do
  let v = A.fromList (Z :. b) (repeat 0)
  m <- Random.randomArray (\_ -> MWC.normal 0 0.01) (Z :. b :. a)
  return (v, m)

-- A single dense neural network layer with sigmoid activations
denseModel :: ParaLens (,) (DSL Acc) DenseParams (Vector Double) (Vector Double)
denseModel = NN.dense NN.sigmoid
