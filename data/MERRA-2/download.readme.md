# Downloading

Instructions from [here](https://disc.sci.gsfc.nasa.gov/data-access#mac_linux_wget)

##wget for Mac/Linux

1. Make sure you have setup your Earthdata account.
1. Install wget if necessary. A version of wget 1.18 complied with gnuTLS 3.3.3 or OpenSSL 1.0.2 or LibreSSL 2.0.2 or later is recommended.
1. Create a .netrc file in your home directory.
    1. cd ~ or cd $HOME
    1. touch .netrc
    1. echo "machine urs.earthdata.nasa.gov login <uid> password <password>" >> .netrc (where <uid> is your user name and <password> is your Earthdata Login password without the brackets)
    1. chmod 0600 .netrc (so only you can access it)
1. Create a cookie file. This file will be used to persist sessions across calls to wget or curl.
    1. cd ~ or cd $HOME
    1. touch .urs_cookies.
    1. Note: you may need to re-create .urs_cookies in case you have already executed wget without valid authentication.
1. Download your data using wget: 
`wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition <url>`
    --auth-no-challenge may not be needed depending on your version of wget
    - <url> is the link that points to a file you wish to download or to an OPeNDAP resource.
    - Your Earthdata password might be requested on the first download
    - If you wish to download an entire directory, such as this example URL, use the following command:
    `wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies -np -r --content-disposition <url>`

To download multiple data files at once, create a plain-text <url.txt> file with each line containing a GES DISC data file URL. Then, enter the following command:

`wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -i <url.txt>`

## Parallel downloading
cat subsets.txt  | tr -d '\r' | xargs -P 4 -n 1 curl -s -LJO -n -c ~/.urs_cookies -b ~/.urs_cookies

# Citation
Global Modeling and Assimilation Office (GMAO) (2015), MERRA-2 statD_2d_slv_Nx: 2d,Daily,Aggregated Statistics,Single-Level,Assimilation,Single-Level Diagnostics V5.12.4, Greenbelt, MD, USA, Goddard Earth Sciences Data and Information Services Center (GES DISC), Accessed: 3 Februrary 2021, 10.5067/9SC1VNTWGWV3

Global Modeling and Assimilation Office (GMAO) (2015), MERRA-2 inst1_2d_lfo_Nx: 2d,1-Hourly,Instantaneous,Single-Level,Assimilation,Land Surface Forcings V5.12.4, Greenbelt, MD, USA, Goddard Earth Sciences Data and Information Services Center (GES DISC), Accessed: 3 Fabruary 2021, 10.5067/RCMZA6TL70BG

parallel citation
Tange, O. (2020, November 22). GNU Parallel 20201122 ('Biden').
  Zenodo. https://doi.org/10.5281/zenodo.4284075
