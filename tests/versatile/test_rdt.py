import pytest

import cait as ai
from cait.versatile.datasources.hardwaretriggered.rdt_file import RDTFile
from cait.versatile.datasources.hardwaretriggered.par_file import PARFile


from ..fixtures import tempdir, RDT_LENGTH

@pytest.fixture(scope="class")
def rdt_files(tempdir):
    data1 = ai.data.TestData(filepath=tempdir.name+'/data1', 
                             duration=RDT_LENGTH,
                             channels=[0, 1],
                             sample_frequency=25000,
                             start_s=13)
    data2 = ai.data.TestData(filepath=tempdir.name+'/data2',
                             duration=RDT_LENGTH,
                             channels=[1],
                             sample_frequency=50000,
                             start_s=20349,
                             include_carriers=False)

    data1.generate()
    data2.generate()

    yield RDTFile(tempdir.name+'/data1.rdt'), RDTFile(tempdir.name+'/data2.rdt')

class TestRDT:
    def test_par(self, tempdir, rdt_files): # rdt_files fixture needed to create par files
        p1 = PARFile(tempdir.name+'/data1.par')
        p2 = PARFile(tempdir.name+'/data2.par')

        assert p1.start_s == 13
        assert p2.start_s == 20349

        assert p1.stop_s == 13 + RDT_LENGTH
        assert p2.stop_s == 20349 + RDT_LENGTH

    def test_constructor(self, rdt_files):
        f1, f2 = rdt_files
        assert f1.sample_frequency == 25000
        assert f2.sample_frequency == 50000

        assert f1.keys == [0, 1, (0,1)]
        assert f2.keys == [1]

    def test_channels(self, rdt_files):
        f1, f2 = rdt_files
        assert f1[0].n_channels == 1
        assert f1[1].n_channels == 1
        assert f1[(0,1)].n_channels == 2

        assert f2[1].n_channels == 1

        tpas = [-1.0, 0.0, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]
        assert [float(str(x)) for x in f1[0].unique_tpas] == tpas
        assert [float(str(x)) for x in f1[(0,1)].unique_tpas] == tpas
        assert [float(str(x)) for x in f2[1].unique_tpas] == tpas

    def test_methods_rdt(self, rdt_files):
        f1, f2 = rdt_files
        # I'M STILL NOT SURE IF I WANT A DEFAULT CHANNELS BEHAVIOR OR NOT
        # assert self.f1.default_channels == (0,1)
        # assert self.f2.default_channels == 1

        f1.dt_us
        f1.measuring_time_h
        f1.keys
        f1.get_trace([0,1,2], voltage=True)
        f1.get_trace([0,1,2], voltage=False)

        f2.dt_us
        f2.measuring_time_h
        f2.keys
        f2.get_trace([0,1,2], voltage=True)
        f2.get_trace([0,1,2], voltage=False)

    def test_methods_channels(self, rdt_files):
        f1, f2 = rdt_files
        f1[0].key
        f1[(0,1)].key
        f2[1].key

        f1[0].timestamps
        f1[(0,1)].timestamps
        f2[1].timestamps

        f1[0].tpas
        f1[(0,1)].tpas
        f2[1].tpas

        it1 = f1[0].get_event_iterator()
        it2 = f1[(0,1)].get_event_iterator()
        it3 = f2[1].get_event_iterator()

        assert it1.n_channels == 1
        assert it2.n_channels == 2
        assert it3.n_channels == 1

        next(iter(it1))
        next(iter(it2))
        next(iter(it3))