.. _thedatahandlerclass:

*********************
The DataHandler Class
*********************

The DataHandler class handles most of the processing and organization of your data. It is heavily implemented, consisting
of the child class DataHandler, which inherits methods from several mixins classes. A mixin is a attribute-less base
with the purpose of providing further methods to the child class, without overloading the code-content of the child
class. The mixins are structured thematically. For the child class, as well as for every mixin class, an individual
documantation page exists.

.. toctree::
   :caption: Child and Mixins
   :maxdepth: 1

   datahandler
   mixins/analysis_mixin
   mixins/csmpl_mixin
   mixins/features_mixin
   mixins/fit_mixin
   mixins/plot_mixin
   mixins/rdt_mixin
   mixins/simulate_mixin
   mixins/bin_mixin
   mixins/trigger_collection_mixin
