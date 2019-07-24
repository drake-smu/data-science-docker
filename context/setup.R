package_list = c(
    'ggplot2', 
    'plyr', 
    'reshape2', 
    'dplyr', 
    'tidyr', 
    'psych', 
    'pwr', 
    'STAR', 
    'ez', 
    'bursts')

# Install Packages ------------------------------
install.packages("devtools")
install.packages("RCurl")
install.packages(package_list)

# Install IRkernel ------------------------------

devtools::install_github('IRkernel/IRkernel') 
IRkernel::installspec()