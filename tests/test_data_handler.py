import pytest

# here are only a few examples how tests are written

def test_example():
    x = 5
    y = 6
    assert x + 1 == y, "test failed"

@pytest.fixture
def supply_AA_BB_CC():
	aa=25
	bb =35
	cc=45
	return [aa,bb,cc]

def test_comparewithAA(supply_AA_BB_CC):
	zz=35
	assert supply_AA_BB_CC[0]==zz,"aa and zz comparison failed"

def test_comparewithBB(supply_AA_BB_CC):
	zz=35
	assert supply_AA_BB_CC[1]==zz,"bb and zz comparison failed"

def test_comparewithCC(supply_AA_BB_CC):
	zz=35
	assert supply_AA_BB_CC[2]==zz,"cc and zz comparison failed"

@pytest.mark.skip
def test_add_2():
	assert 100+200 == 300,"failed"

@pytest.mark.xfail
def test_add_3():
	assert 15+13 == 28,"failed"

# tests can be added below

