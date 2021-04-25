###################################
#       Data processing file
###################################


"""
You should download some necessary modules using the below comand
lines if you don't have theme already on your computer :

# Execute these lines on your Terminal :
$ pip install git+https://github.com/spacesense-ai/spacesense.git
$ pip install sentinelsat==0.13
$ pip install rasterio
$ pip install sklearn

If osgeo module (gdal) isn't already on your computer or it dosen't work:
    You can check this link : https://pypi.org/project/GDAL/
    Or simply execute this comande : $ sudo easy_install GDAL 
"""


# Scientific and Graphic libraries
import numpy as np  # algebre linear
import pandas as pd  # traitement de données
from geojson import Polygon, Feature, FeatureCollection, dump, load
import geojson
from astropy.visualization import make_lupton_rgb
np.seterr(invalid='ignore')
import matplotlib.pyplot as plt


# General libraries
from pathlib import Path
from datetime import date, timedelta
import datetime
import time
import pandas as pd
import os
import shutil
import requests


# Space sence libraries
import spacesense
from spacesense.datasets import download_sentinel
from spacesense import datasets, utils


# Upload fire location and date from  a csv file
df1 = pd.read_csv('../fireData/archive/fire_nrt_V1_101674.csv')
df2 = pd.read_csv('../fireData/archive/fire_archive_V1_101674.csv')
df_merged = pd.concat([df1, df2], sort=True)


df_new = df_merged[["latitude", "longitude", "acq_date", "frp"]]
df = df_new[df_new['acq_date'] >= '2019-11-01']


# Sort our data to most affected zone
df_topaffected = df.sort_values(by='frp', ascending=False)


class Fire_spot(object):
    """ Compute and save fire files """

    # Copernicus account information
    USER = 'mohamedabdallahimohamed'
    PASSWORD = 'zywjys-cetje7-hytTiw'

    def __init__(self, lat=None, lng=None, date=None, label=None, radius=1.5, time_search_width=40):
        self.lat = lat
        self.lng = lng
        self.date = date
        self.name = label
        self.radius = radius
        self.bfr_filename = None
        self.aftr_filename = None
        self.download_cursor = download_sentinel(
            Fire_spot.USER, Fire_spot.PASSWORD)
        self.aoi = None
        self.deltatime = datetime.timedelta(days=time_search_width)

    def sqrt_geometry(self, radius):
        """ Creat a polygon using a centrale point and radius """

        square_coordinate = [(self.lng - radius, self.lat - radius),
                             (self.lng - radius, self.lat + radius),
                             (self.lng + radius, self.lat + radius),
                             (self.lng + radius, self.lat - radius),
                             (self.lng - radius, self.lat - radius)]

        return square_coordinate

    def creat_json(self):
        """ Save the square as geoJASON file """

        polygon = self.sqrt_geometry(self.radius*0.015)
        p1 = Polygon([polygon])
        my_feauter3 = Feature(geometry=p1, properties={})
        feature_collection = FeatureCollection([my_feauter3])
        self.aoi = '../geoJSONfiles/'+self.name+'.geojson'
        with open(self.aoi, 'w') as f:
            f.write(geojson.dumps(feature_collection))

    def get_product(self):
        """
        Download  all bands files from befor and after fire event

        TODO : implement is_avalible and download_product functions
        """

        # Creat geojson file to our spot
        self.creat_json()

        bfr_key = self.get_product_key("befor")
        aft_key = self.get_product_key("after")

        if (bfr_key != False and aft_key != False):
            print("[ok]")
            print("downloading the befor event file :")
            self.download_product("befor", bfr_key)

            print("downloading the after event file :")
            self.download_product("after", aft_key)
            return True

        else:
            print(" there is no product for this event")
            return False

    def download_product(self, event_position, prod_key):
        """ Download a product by its Key """

        # TODO : implement this function that download this articls
        # in ../downloaded_images/self.name

        if(not Path("../downloaded_images/"+self.name).exists()):
            Path("../downloaded_images/"+self.name).mkdir()

        if(Path("../downloaded_images/"+self.name+"/"+event_position).exists()):
            print(" This file already exist")

        else:
            Path("../downloaded_images/"+self.name+"/"+event_position).mkdir()
            self.download_cursor.download_files([prod_key],
                                                directory_path="../downloaded_images/"+self.name+"/"+event_position)

    def get_product_key(self, event_position):
        """ Return the Key of the product with less cloud cover and available online """

        if (event_position == "befor"):
            self.download_cursor.sentinel_2(roi_polygon=self.aoi, startdate=self.date + timedelta(days=-7) - self.deltatime,
                                            enddate=self.date + timedelta(days=-7), cloudcover_max=4)
            return self.is_available("befor")

        elif (event_position == "after"):

            # wait 10s to not have connexion issue with the Server
            time.sleep(10)
            self.download_cursor.sentinel_2(roi_polygon=self.aoi, startdate=self.date + timedelta(days=7),
                                            enddate=self.date + timedelta(days=7) + self.deltatime, cloudcover_max=4)
            return self.is_available("after")

    def is_available(self, event_position):
        """ Check  if the product is 'Online' on copernicus """
        product_number = len(self.download_cursor.list_products)

        online_list = []
        for i in range(product_number):
            is_online = self.download_cursor.api.get_product_odata(
                self.download_cursor.list_products[i][0])["Online"]
            if(is_online):
                online_list += [(i, self.download_cursor.list_products[i]
                                 [1]['cloudcoverpercentage'])]

        indx = Fire_spot.chose_index(online_list)

        if(indx == None):
            return False
        else:
            if (event_position == "befor"):
                self.bfr_filename = self.download_cursor.list_products[indx][1]["filename"]
            elif(event_position == "after"):
                self.aftr_filename = self.download_cursor.list_products[indx][1]["filename"]
            return self.download_cursor.list_products[indx][0]

    @staticmethod
    def chose_index(cloud_tab):
        """ Chose the index of a product with less cloud cover """

        # Debug
        print(" \ncloud cover tab :")
        print(cloud_tab)

        if len(cloud_tab) == 0:
            return
        else:
            min_indx = 0
            for i in range(1, len(cloud_tab)):
                if (cloud_tab[i][1] < cloud_tab[min_indx][1]):
                    min_indx = i
            return cloud_tab[min_indx][0]


class Image_processing(object):
    """ Class to process images and calculat spectral indices """

    def __init__(self, fire_spot=Fire_spot()):

        self.name = fire_spot.name
        self.bfr_file_path = "../downloaded_images/" + \
            self.name+"/befor/"+fire_spot.bfr_filename
        self.aftr_file_path = "../downloaded_images/" + \
            self.name+"/after/"+fire_spot.aftr_filename
        self.aoi = '../geoJSONfiles/'+self.name+'.geojson'

        if(not (Path(self.bfr_file_path).exists()
                and Path(self.aftr_file_path).exists()
                and Path(self.aoi).exists())):

            raise TypeError(" This fire spot dosen't exist ")

        else:
            self.bfr_image_cursor = datasets.read_sentinel_2(
                folder_path=self.bfr_file_path)
            self.aftr_image_cursor = datasets.read_sentinel_2(
                folder_path=self.aftr_file_path)
            self.bands_bfr = None
            self.bands_aftr = None

    def fetch_bands(self):
        """ put bands in the data frame """

        _, data_resampled = self.bfr_image_cursor.get_data(
            AOI=self.aoi, resample=True, resize_raster_source='B02')

        band_names = sorted(self.bfr_image_cursor.data_dictionary.keys())

        self.bands_bfr = {}
        for i in range(len(band_names)):
            self.bands_bfr.update({band_names[i]: data_resampled[i]})

        _, data_resampled = self.aftr_image_cursor.get_data(
            AOI=self.aoi, resample=True, resize_raster_source='B02')

        band_names = sorted(self.aftr_image_cursor.data_dictionary.keys())

        self.bands_aftr = {}
        for i in range(len(band_names)):
            self.bands_aftr.update({band_names[i]: data_resampled[i]})

        print("prossess end")

    def calc_indices(self):
        """ 
        Calculate all indices we need and update the dictionary data
        """

        # Befor the incident time
        self.bands_bfr["NDVI"] = Image_processing.ndvi_index(self.bands_bfr)
        self.bands_bfr["NDMI"] = Image_processing.ndmi_index(self.bands_bfr)
        self.bands_bfr["NBRI"] = Image_processing.nbri_index(self.bands_bfr)
        self.bands_bfr["TCI"] = Image_processing.true_color_img(self.bands_bfr)

        # After the incident time 
        self.bands_aftr["NBRI"] = Image_processing.nbri_index(self.bands_aftr)
        self.bands_aftr["TCI"] = Image_processing.true_color_img(self.bands_aftr)

        # burned zone
        self.bands_bfr["dNBRI"] = self.bands_bfr["NBRI"] - self.bands_aftr["NBRI"]

    def save_result(self):
        """ Save the result of calculation in a csv file """

        if(not Path('../processing_result/'+self.name).exists()):
                Path('../processing_result/'+self.name).mkdir()

        np.savetxt('../processing_result/'+self.name+'/vegetation_density.csv', self.bands_bfr["NDVI"], delimiter=',')
        np.savetxt('../processing_result/'+self.name+'/humidity.csv', self.bands_bfr["NDMI"], delimiter=',')
        np.savetxt('../processing_result/'+self.name+'/burned_area.csv', self.bands_bfr["dNBRI"], delimiter=',')


    @staticmethod
    def ndvi_index(data):
        """
        Normalized difference vegetation index
        NDVI = (b_nir - b_red)/(b_nir + b_red)
        :return: NDVI values for each pixel
        """

        red = data["B04"]
        nir = data["B08"]
        return np.where((nir+red) == 0.0, 0, (nir-red)/(nir+red))

    @staticmethod
    def ndmi_index(data):
        """
        Normalized difference moisture index
        It is used to determine vegetation water content. NDMI calculated as a ratio between 
        the NIR and Short Wave Infra-Red (SWIR) values in traditional fashion:


        NDWI = (b_nir - b11_swir)/(b_nir + b11_swir)
        :return: NDMI values for each pixel
        """

        nir = data["B08"]
        swir = data["B11"]
        return np.where(((nir + swir) == 0.0), 0, (nir - swir)/(nir + swir))

    @staticmethod
    def nbri_index(data):
        """
        Normalized burn ratio index
        It uses the NIR and SWIR2 channels to highlight burned areas.


        NBRI = (b_nir - b12_swir)/(b_nir + b12_swir)
        :return: NBRI values for each pixel
        """

        nir = data["B08"]
        swir = data["B12"]
        return np.where(((nir + swir) == 0.0), 0, (nir - swir)/(nir + swir))

    @staticmethod
    def true_color_img(data):
        blue = data["B02"]
        green = data["B03"]
        red = data["B04"]

        r = red/150
        b = blue/150
        g = green/150
        TCimage = make_lupton_rgb(r, g, b, Q=4) # merge the three colors
        return TCimage



##########################
# THE END OF THIS FILE
##########################


def fetch_exemple(idx):

    lat = float(df_topaffected.iloc[[idx]]['latitude'])
    lng = float(df_topaffected.iloc[[idx]]['longitude'])

    date_aq = df_topaffected.iloc[[idx]]['acq_date'].values[0]
    date_time_obj = datetime.datetime.strptime(date_aq, '%Y-%m-%d')
    date_instance = date_time_obj.date()

    return {"lat": lat, "lng": lng, "date_aq": date_instance}


def get_df():
    return df_topaffected

def plot_result(spec_image, Save=True):
    imgs_num = 2
    col_num = 2
    plt.figure(figsize=(30,40))

    # Img 1

    plt.subplot(int(col_num/imgs_num) + 1 , col_num,  1)
    plt.title("Befor the fire event")
    plt.imshow(spec_image.bands_bfr["TCI"])

    #Img 2
    plt.subplot(int(col_num/imgs_num) + 1, col_num,  2)
    plt.title("After the fire event")
    plt.imshow(spec_image.bands_aftr["TCI"])

    #Img3
    plt.subplot(int(col_num/imgs_num) + 1 , col_num, 3)
    plt.title("Maping the burnt zone")
    trueImageCUT = spec_image.bands_bfr["TCI"].copy()
    red  = np.array([255, 70, 0],dtype=np.uint8)
    burnCUT = np.where( spec_image.bands_bfr["dNBRI"]>0.5 , True ,False )
    trueImageCUT[burnCUT] = red
    plt.imshow(trueImageCUT)

    #Img4
    plt.subplot(int(col_num/imgs_num) + 1 , col_num, 4)
    plt.title("Illustration of the burn area")
    burnCUT = np.where( spec_image.bands_bfr["dNBRI"]>0.5 , spec_image.bands_bfr["dNBRI"] ,0 )
    plt.imshow(burnCUT, cmap = 'hot' )

    if (Save == True):
        # Save our image
        if(not Path('../processing_result/'+spec_image.name).exists()):
                Path('../processing_result/'+spec_image.name).mkdir()
        plt.savefig('../processing_result/'+spec_image.name+'/visual_inf.png')

def write_num(i):
    with open('../other/last_step.txt', 'w') as f:
      f.write('%d' % i)

def read_num():
    with open('../other/last_step.txt', 'r') as f:
      i = int(f.readline())
    return i

def rmv_download_img(i):
    """delete image with index i from downloaded_images folder"""

    if os.path.exists("../downloaded_images/Australia_fire_"+str(i)):
       shutil.rmtree("../downloaded_images/Australia_fire_"+str(i))
       print("Done")
    else:
      print("The file does not exist")

def rmv_geojson(zone):
    """delete geojson with index i from downloaded_images folder"""

    if os.path.exists(zone.aoi):
       os.remove(zone.aoi)
       print("Done")
    else:
      print("The file does not exist")

def get_coordonates(zone): 
    """ get the coordonates from the geo json file """

    path_to_file = zone.aoi
    if os.path.exists(path_to_file):
        with open(path_to_file) as f:
            gj = geojson.load(f)
            features = gj['features'][0]["geometry"]["coordinates"][0]
    return features
        
def is_water(lat, lng):
    """ Send a request to know if this point is in water place """

    # api-endpoint
    URL = "https://api.onwater.io/api/v1/results/"+str(lat)+","+str(lng)+"?access_token=2Ht7p3Gf6uuXwj28nW4h"
    print(URL)
    try : 
        r = requests.get(url = URL)
        # extracting data in json format
        data = r.json()
        return data["water"]
    except :
        print ("Was not founded")
        return False

def there_is_water(zone):
    coords = get_coordonates(zone)
    check = False
    for coord  in coords : 
        time.sleep(4)
        request = is_water(coord[1], coord[0])
        check = check or request
    if check :
        print("There is water" )
    else : 
        print("There is no water" )      
    return check

def dist(i, j):    
    return np.sqrt(sum((df_topaffected.iloc[i][0:2]-df_topaffected.iloc[j][0:2])**2))/0.015

def all_dowloaded_indices():
    l = os.listdir("../processing_result/")
    l.remove('.DS_Store')
    t = [int(s.split("_")[-1]) for s in l ]
    t.sort()
    return t

def is_repeated(idx):
    array = all_dowloaded_indices()
    for i in array :
        if dist(i, idx) < 0.7:
            print("Like {}".format(i))
            return True
    return False



def run(): 
    i = read_num()
    rmv_download_img(i)
    short_name = "Australia_fire"
    while(True):
        try :
            # try this
            print(" \n\n Try the spot "+ str(i) + " :\n")
            if is_repeated(i):
                # This spot was downloaded befor
                print("Skip it")
            else :
                # Wasn't downloaded befor
                spot_cordonate = fetch_exemple(i)
                zone = Fire_spot(lat = spot_cordonate['lat'], lng = spot_cordonate['lng'], date = spot_cordonate['date_aq'], label = short_name+"_"+str(i), radius=1.5)
                zone.creat_json()
                if(there_is_water(zone)) : 
                    # There is water in this spot
                    rmv_geojson(zone) # delet the geojson file
                else :
                    # There no water in this pot
                    if(zone.get_product()):
                        print("\n---------------\n\nstart of processing")
                        spec_image = Image_processing(zone)
                        spec_image.fetch_bands()
                        spec_image.calc_indices()
                        spec_image.save_result()
                        print("\n---------------\n\nend processing\n")
                        plot_result(spec_image)
                        rmv_download_img(i)  # Clean dowloaded_images file from data that we don't need

                    else :
                        # TODO : delete the geojson file 
                        rmv_geojson(zone) # delet the geojson file
        
            i+=1
            write_num(i)
            time.sleep(10)
        except :
            print("There is an Error ")
            i+= 1
run()