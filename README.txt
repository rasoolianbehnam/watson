----------------------------------
DIRECTORY STRUCTURE:
----------------------------------
----
data: 
----
This is were all data are stored/saved.
run get_data.sh in order to automatically
download and extract data needed in order
to start the project. 
These data include:
    - contact_maps: The raw full contact maps
    - edges: contact results of thresholding contact maps.
    - graphlets: extracted orbits from edges.
    - orbits: orbits ordered so they can be served as input for correlation procedures
    - loci
-------
external:
-------
Contains scripts and binaries from other languages
than python that are required in this project.
The main ones include the 'orca' package in R
programming language that can be installed
using install.packages('orca') in R.
Also a java jar file is used to calculate
MIC values from orbit information.

---------
notebooks
---------
Contains jupyter notebooks for a demo of
what we did in the project

----------------
requirements.txt
----------------
run 'pip -r requirements.txt' in order to
install all pre-requisite packages in python.
Also, you need to install iced package 
manually in python, which can be downloaded
from here:
https://graphiclet.localtunnel.me/secret/iced.zip
