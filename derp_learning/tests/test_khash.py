import pytest
from derp_learning.util import jit
# from derp_learning.khash import KHashi8i8
# from derp_learning.khash.khash_cffi import _khash_get
import numpy as np

@jit
def use_khash_jitclass(h, i):
   return h.get(i) + 10

@pytest.mark.skip('numba cffi broken?')
def test_khash_jitclass():
   h = KHashi8i8()
   h.update([(7, 3), (13, 4)])
   h.set(1, 13)
   assert h.get(1) == 13
   assert h.get(7) == 3
   assert h.get(13) == 4
   assert use_khash_jitclass(h, 1) == 23
   assert use_khash_jitclass(h, 7) == 13
   assert use_khash_jitclass(h, 13) == 14
   assert h.get(-2345) == -123456789
   assert h.get(926347) == -123456789
   assert h.size() == 3

@jit
def numba_get(h, i):
   return h.get(i)

def foo(h):
   hash = h.hash

   @jit
   def func(i):
      return _khash_get(hash, i, -123456789)

   return func

@pytest.mark.skip('numba cffi broken?')
def test_khash_numba_closure():
   h = KHashi8i8()
   h.set(10, 10)

   assert numba_get(h, 10) == 10

   f = foo(h)
   assert f(10) == 10

@pytest.mark.skip('numba cffi broken?')
def test_khash_array_get():
   h = KHashi8i8()
   a = np.arange(10)
   h.update2(a, a + 1)
   ahat = h.array_get(a)
   assert np.all(ahat == a + 1)
