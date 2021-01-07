{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
module Main where

import Prelude hiding (id)
import Control.Monad
import Data.Array.Accelerate (Acc, Exp, Matrix, Vector, Scalar, Elt, use, fromList, Z(..), (:.)(..), pattern I1, pattern I2, (!), pattern T2)
import qualified Data.Array.Accelerate as A
--import qualified Data.Array.Accelerate.LLVM.PTX as PTX
--import qualified Data.Array.Accelerate.LLVM.Native as Native
import qualified Data.Array.Accelerate.LLVM.PTX as Backend

import Control.Categories ((~>))
import Numeric.Optics (MonoLens(..), runDSL)
import Numeric.Optics.Base.Accelerate (projection, reshapeLens)
import qualified Numeric.Optics.Base.Accelerate.NeuralNetwork as NN

import System.Random.Shuffle (shuffleM)

import Iris
import Training

-- try out a few different example models with gradient descent
main :: IO ()
main = do
  testBaseline
  testBaselineOneHot
  testBaselineLensModel
  testScaleModel
  testScaleDense

  putStrLn "\n\n\n"
  void $ replicateM 10 testDense

-------------------------------
-- Tests

testBaseline :: IO ()
testBaseline = do
  d <- irisData
  let a        = use (toAccelerateData d)
      b        = use (toAccelerateLabels d)
      
  -- 86% accuracy with baseline model
  putStrLn "==============================="
  putStrLn "Baseline accuracy"
  print . Backend.run . A.unit . accuracy $ correctlyClassified baselineModel a b
  putStrLn "===============================\n\n"

testBaselineOneHot :: IO ()
testBaselineOneHot = do
  d <- irisData
  let a = use (toAccelerateData d)
      b = use (toAccelerateLabelsOneHot d)
      
  -- 86% accuracy with baseline model
  putStrLn "==============================="
  putStrLn "Baseline (one-hot) accuracy"
  print . Backend.run . A.unit . accuracy $ correctlyClassified baselineModelOneHot a b
  putStrLn "===============================\n\n"


-- The baseline model, expressed as a ParaLens with parameters @()@.
testBaselineLensModel :: IO ()
testBaselineLensModel = do
  d <- irisData
  let a = use (toAccelerateData d)
      b = use (toAccelerateLabels d)
      (I2 n _) = A.shape b

      MonoLens f f' = step (gdUpdate (0.01 :: Exp Double)) meanSquaredErrorDisplacement baselineModelLens
      fwd p a   = runDSL f (T2 p a)
      rev p a b = Backend.run1 (A.afst . runDSL f') ((p, a), b)

  let ps = trainVector rev () a b
  let !pFinal = last ps

  putStrLn "==============================="
  putStrLn "Baseline (lens) accuracy"
  let trainedModel = (\x -> fwd (use pFinal) x)
  print . Backend.run . A.unit . accuracy $ correctlyClassified trainedModel a b
  putStrLn "===============================\n\n"


testScaleModel :: IO ()
testScaleModel = do
  d <- irisData
  let a = use (toAccelerateData d)
      b = use (toAccelerateLabels d)
      (I2 n _) = A.shape b

      MonoLens f f' = step (gdUpdate (0.01 :: Exp Double)) meanSquaredErrorDisplacement scaleModel
      fwd p a   = runDSL f (T2 p a)
      rev p a b = Backend.run1 (A.afst . runDSL f') ((p, a), b)

  let ps = trainVector rev (Backend.run scaleModelParams) a b
  let pFinal = last ps

  putStrLn "==============================="
  print pFinal
  putStrLn "testScaleModel accuracy"
  let trainedModel = (\x -> fwd (use pFinal) x)
  print . Backend.run . A.unit . accuracy $ correctlyClassified trainedModel a b
  putStrLn "===============================\n\n"

testScaleDense :: IO ()
testScaleDense = do
  d <- irisData
  let a = use (toAccelerateData d)
      b = use (toAccelerateLabels d)
      (I2 n _) = A.shape b

      MonoLens f f' = step (gdUpdate (0.01 :: Exp Double)) meanSquaredErrorDisplacement scaleDense
      fwd p a   = runDSL f (T2 p a)
      rev p a b = Backend.run1 (A.afst . runDSL f') ((p, a), b)

  params <- mkDenseParams 1 1
  let ps = trainVector rev params a b
  let pFinal = last ps

  putStrLn "==============================="
  print pFinal
  putStrLn "testScaleDense accuracy"
  let trainedModel = (\x -> fwd (use pFinal) x)
  print . Backend.run . A.unit . accuracy $ correctlyClassified trainedModel a b
  putStrLn "===============================\n\n"

argMaxOneHot :: (A.Eq a, A.Ord a) => Acc (Vector a) -> Acc (Vector Double)
argMaxOneHot x = A.map (\a -> (a A.== A.the m) A.? (1.0, 0.0)) x
  where m = A.maximum x

testDense :: IO ()
testDense = do
  let numEpochs = 10
  -- Create @numEpochs@ copies of irisData, each shuffled differently.
  -- This is to imitate the behaviour of Keras, where the shuffle=True flag is
  -- used to shuffle the dataset before each training epoch.
  d <- fmap concat . sequence . replicateM numEpochs shuffleM =<< irisData

  let a = use (toAccelerateData d)
      b = (A.map A.fromIntegral . use $ toAccelerateLabelsOneHot d) :: Acc (Matrix Double)

      MonoLens f f' = step (gdUpdate (0.01 :: Exp Double)) meanSquaredErrorDisplacement denseModel
      fwd p a   = argMaxOneHot $ runDSL f (T2 p a)
      rev p a b = Backend.run1 (A.afst . runDSL f') ((p, a), b)

  params <- mkDenseParams 4 3
  let ps = trainVector rev params a b
  let pFinal = last ps
  putStrLn "==============================="
  putStrLn "testDense"
  putStrLn "final parameters "
  print pFinal

  putStrLn "training accuracy:"
  print . Backend.run . A.unit . accuracy $ correctlyClassified (fwd $ use pFinal) a b
  putStrLn "===============================\n\n"

-------------------------------
-- TODO: parametrise train functions by a backend and move to Train.hs
-- @run :: Acc a -> a@

-- | Train a model using a function which updates parameters from a given
-- example @(x, y) :: (a, b)@
train :: ()
  => (p -> a -> b -> p) -- ^ model update
  -> p  -- ^ initial params
  -> [(a, b)] -- ^ dataset
  -> [p]
train f = Prelude.scanl (\p (a, b) -> f p a b)

-- | Train a model where the dataset is stored as a pair of matrices (i.e.,
-- inputs and outputs are 'Vector's
trainVector :: forall p a b . (Elt a, Elt b)
  => (p -> Vector a -> Vector b -> p)
  -> p
  -> Acc (Matrix a)
  -> Acc (Matrix b)
  -> [p]
trainVector f p0 a b = train f p0 abs where
  abs = map (\i -> (getA i, getB i)) [0..n - 1]
  n = runExp (numRows a)

  getA :: Int -> Vector a
  getA i = Backend.run . A.slice a . A.lift $ Z :. i `mod` n :. A.All

  getB :: Int -> Vector b
  getB i = Backend.run . A.slice b . A.lift $ Z :. i `mod` n :. A.All

numRows :: Elt a => Acc (Matrix a) -> Exp Int
numRows a = let (I2 n _) = A.shape a in n

runExp :: Elt e => Exp e -> e
runExp e = A.indexArray (Backend.run (A.unit e)) Z
