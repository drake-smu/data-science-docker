package_list = c(
    'devtools', 
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

install.packages(package_list)

# Install IRkernel ------------------------------

devtools::install_github('IRkernel/IRkernel') 
IRkernel::installspec()