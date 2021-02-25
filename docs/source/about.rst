*****
About
*****

We started Cait (Cryogenic Artificial Intelligence Tools) as a Python project in 2020, to include machine learning methods in the analysis process of the cryogenic dark matter experiments CRESST and COSINUS. Due to the numerous available Python packages for scientific computing, it is also suitable for prototyping of new algorithms and analysis methods. Thanks to the efficient implementation of high-performance computing packages as Numpy and Scipy, Cait competes well for heavy lifting computing tasks as the triggering and large scale processing of raw data. The IPython functionality and integration in Jupyter Notebooks makes the work on remote vizualization clients of HPC clusters (MPCDF, CLIPP, ...) especially convenient.

We follow a set of paradigms in the development of Cait, that ensure attractiveness and long-term support of our software:

- **Integrability:** Due to the development as a Python3 package, rather than as a closed software, the integration of other Python packages is just a line of code away. This especially enables the connection to cutting-edge machine learning libraries, as Scikit-Learn and Pytorch Lightning.

- **Usability:** With a variety of tutorial notebooks and an user-friendly API we open the door for users with limited programming skills in Python. A standard analysis for small and ordinary data files is doable with out-of-the-box instructions, in little more than an hour from data conversion to publication-ready plots.

- **Sustainability:** We follow clear code and documentation guidelines to lower the threshold for first contributions. Many developers and users are students or scientists with little formal training in software development. The guidelines improve the readability of source code and make the time efficient implementation of your own ideas possible.

*Fun Fact:* The name Cait is short for Caitlin, an originally irish name, anglicised and spelled as "Kayt-lin". There are several reasons, why a human name and Cait in particular suits our software package very well: The name of the main analysis package within the CRESST collaboration is CAT (Cryogenic Analysis Tools), to which Cait is an extension in some sense. Also, our package for dark matter limit calculations is called Romeo (written with the programming language Julia). Last, but not least, it was a tradition within CRESST to name phonon detectors with female and light detectors with male names (Frederika, Lise, Michael, ...).