*******************
Bandfit Parameters
*******************

For instanciating the Bandfit class, we need to hand starting values, bounds and fix/free flags for all parameters. For
this we include here some suitable values for some detectors. Values for new detectors can be modelled after these values.

CRESST, Det A
==============

.. code:: python

    names_module_independent = ["np_decay", "np_fract", "L0", "L1", "sigma_l0", "S1", # these are just for information
                                "S2", "el_amp", "el_decay", "el_width",  # do not put as argument to method!
                                "sigma_p0", "sigma_p1", "E_p0", "E_p1", "E_fr", "E_dc",
                                "L_lee", "QF_y", "QF_ye", "eps", "kg_d", "thr"]

    values_module_independent = [1.78139, 0.716578, 0.888663, 0.0113157, 0.094, 0.234057,
                                 0.0208574, 62.0497, 10.5649, 7.37096, 0.0045,
                                 0.010067, 81.8496, -1.40864, 2.42103e5, 0.0339261, 0.0111147,
                                 0.972807, -0.00884255, 0.711406, 5.689, 0.0301]

    names_nuclear = [["QF_cawO", "es_cawO_f", "es_cawO_lb", "nc_O_p0", "nc_O_p1"],  # these are just for information
                    ["QF_CAwo", "es_CAwo_f", "es_CAwo_lb", "nc_Ca_p0", "nc_Ca_p1"],  # do not put as argument to method!
                    ["QF_caWo", "es_caWo_f", "es_caWo_lb", "nc_W_p0", "nc_W_p1"]]

    values_nuclear = [[0.0739, 0.7088, 567.1, 84.0283, 41.324],
                        [0.0556, 0.1887, 801.3, 134.388, 15.0951],
                        [0.0196, 0.0, 1.0e6, 1643.2, 1.4758]]

    lbounds_nuclear = [[0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0]]

    ubounds_nuclear = [[1.0, 1.0, 1.0e6, 1000.0, 300.0],
               [1.0, 1.0, 1.0e6, 2000.0, 300.0],
               [1.0, 1.0, 1.0e6, 5000.0, 300.0]]

    fixed_nuclear = [[1, 1, 1, 0, 0],
             [1, 1, 1, 0, 0],
             [1, 1, 1, 0, 0]]

    names_gamma = [["FG_1_C", "FG_1_M"],
                   ["FG_2_C", "FG_2_M"],
                   ["FG_3_C", "FG_3_M"],
                   ["FG_4_C", "FG_4_M"],
                   ["FG_5_C", "FG_5_M"]]

    values_gamma = [[15.2208, 2.62553],
                    [22.2517, 7.96598],
                    [28.6565, 10.9598],
                    [54.6522, 11.2899],
                    [19.2077, 2.79869]]

    lbounds_gamma = [[0.0, 2.5], [0.0, 7.9], [0.0, 10.6], [0.0, 11.1], [0.0, 2.7]]

    ubounds_gamma = [[50.0, 2.7], [50.0, 8.1], [50.0, 11.0], [100.0, 11.4], [50.0, 2.9]]

    fixed_gamma = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]

CRESST, TUM 40
================

Coming soon...
