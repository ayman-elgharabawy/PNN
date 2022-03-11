import numpy as np
from sympy import *
import pandas as pd 


def getrank(stringval):
    rankmap = {
		# "a" : 0,"b" : 0,"c" : 0,"d" : 0,"e" : 0,"f" : 0,"g" : 0,"h" : 0,"i" : 0,"j" : 0,"k" : 0,"l" : 0,"m" : 0,"n" : 0,"o" : 0,"p" : 0,"q" : 0,"r" : 0,"s" : 0,"t" : 0,"u" : 0,"v" : 0,
        # "w" : 0,"x" : 0,"y" : 0,"z" : 0
	}
    ranklist=list()
    templist=list()
    keyindexlist=list()
    counter=1
    for i in range(len(stringval)):
        if(stringval[i]!='<' and stringval[i]!='>'):
         xx=stringval[i]
         templist.append(xx)
        else:
         xxx=','.join(templist) 
         rankmap[xxx]=counter
         templist[:] = []
         counter=counter+1
    xxx=','.join(templist) 
    rankmap[xxx]=counter
    templist[:] = []

    labels=[]
    for c in ['a','b','c','d','e','f','g']:
        for i in range(len(rankmap)):
           ll= list(rankmap)[i]
           if c in ll:
               labels.append(i+1)

    return labels


def labelsprocessing(labelsarray):
    ranklist=list()
    for i in  labelsarray:
       row1=getrank(i)
       ranklist.append(row1)    
    return ranklist



def loadDatapermutation(filename):
    filename ='C:\\Ayman\PhDThesis\\' + filename + '.txt'
    file_handler = open(filename, "r")  
    data = pd.read_csv(file_handler, sep = ",") 
    file_handler.close() 

    #############algea###############################
    gender = {'M': 1,'F': 2} 
    city = {'Modesto': 1,'Royal Oak': 2,'San Francisco': 3,'San Diego': 4,'Tamuning': 5,'Atlanta': 6,'Alexandria': 7,'Wichita': 8,'Thousand Oaks': 9
    ,'Columbia': 10,'Phoenix': 11,'Saratoga Springs': 12,'Anaheim': 13,'Minneapolis': 14,'Bayonne': 15,'Cincinnati': 16,'Bronx': 17,'Boise': 18
    ,'Madison': 19,'Saint Paul': 20,'Ithaca': 21,'Southport': 22,'Lawrence': 23,'Chula': 24,'Denver': 25,'Champaign': 26,'Waterville': 27,'New York': 28
    ,'Boston': 29,'Little Rock': 30,'Oshkosh': 31,'Anoka': 32,'Granville': 33,'Eau Claire': 34,'Hopkins': 35,'Arlington': 36,'Chapel Hill': 37,'Jacksonville': 38
    ,'Houston': 39,'Warrenville': 40,'Dallas': 41,'Bordentown': 42,'Kensington': 43,'Lafayette': 44,'Falls Church': 45,'Port Saint Lucie': 46,'Silver Creek': 47,'Mesa': 48
    ,'Stephenville': 49,'Lawton': 50,'San Jose': 51,'El Paso': 52,'Seattle': 53,'Ashland': 54,'Lewiston': 55,'Freedom': 56,'Lakewood': 57,'Charlottesville': 58
    ,'Llano': 59,'Concrete': 60,'Harrisonburg': 61,'Escondido': 62,'Chelsea': 63,'La Habra': 64,'Stephenville': 65,'Edison': 66,'Athens': 67,'Morganton': 68
    ,'Oceanside': 69,'Brookfield': 70,'Kihei': 71,'Ypsilanti':72,'Gainesville': 73,'Staten': 74,'Brooklyn': 75,'Chicago': 76,'Brookfield': 77,'Norman': 78
    ,'Albany': 89,'West Des Moines': 90,'Glen Gardner':91,'Deep River': 92,'Ann Arbor': 93,'Philadelphia':94,'Portland':95,'Davis': 96,'Crystal Lake': 97,'Waterloo': 98
    ,'Fort Wayne': 99,'Redondo Beach': 100,'South Pasadena': 101,'Bowie': 102,'Northridge': 103,'Loma Linda': 104,'Beaverton': 105,'Toledo': 106,'Newport News': 107,'Fairfax': 108
    ,'Charlotte': 109,'Allentown': 110,'Hixson': 111,'Liberty': 112,'Kenmore': 113,'Schaumburg': 114,'Milford': 115,'Niagara Falls': 116,'Mountain View': 117,'Baldwin Park': 118
    ,'Schenectady': 119,'Wilmington': 120,'Fairfield': 121,'Wykagyl': 122,'Sunderland': 123,'San Carlos': 124,'Annapolis': 125,'Fort Collins': 126,'Harrisburg': 127,'Los Angeles': 128
    ,'Sebastopol': 129,'New Haven': 130,'Rancho Santa Margarita': 131,'Plainview': 132,'Delmar': 133,'Evanston': 134,'Livingston': 135,'Grimes': 136,'Huntington Beach': 137,'Rochester': 118
    ,'New Castle': 129,'Mentor': 130,'Danbury': 131,'Santa Rosa': 132,'Oakland': 133,'Alamo': 134,'Pittsburgh': 135,'Turlock': 136,'Sacramento': 137,'Edmond': 138
    ,'Bayamon': 139,'Mesquite': 140,'Bothell': 141,'Meriden': 142,'New Orleans': 143,'Berkeley': 144,'Vienna': 145,'Gaithersburg': 146,'Ann Arbor': 147,'North Grafton': 148
    ,'Palo Alto': 149,'Bristol': 150,'Dubuque': 151,'Los Altos': 152,'Fremont': 153,'Muncie': 154,'Cedar Rapids': 155,'Somerville': 156,'Carol Stream': 157,'Middletown': 158
    ,'Tucson': 159,'Corvallis': 160,'Colorado Springs': 161,'Lexington': 162,'La Jolla': 163,'Grand Rapids': 164,'Renton': 165,'Menlo Park': 166,'Shawnee Mission': 167,'Round Lake': 168
    ,'Columbus': 169,'State College': 170,'Lake Forest': 171,'Providence': 172,'Scituate': 173,'Redwood City': 174,'Malden': 175,'Menlo Park': 176,'Roseburg': 177,'W Hartford': 178
    ,'Campbell': 179,'Owings Mills': 180,'Lompoc': 181,'Marina Del Rey': 182,'Boulder': 183,'West Hollywood': 184,'Cambria': 185,'East Fairfield': 186,'Youngstown': 187,'Mays Landing': 188
    ,'Boca Raton': 189,'Midland': 190,'Canton': 191,'Los Gatos': 192,'West Covina':193,'Fullerton': 194,'Point Pleasant Beach': 195,'Fort Worth': 196,'Fort Worth': 197,'Cambridge': 198
    ,'West Jordan': 199,'Garden City': 200,'Owosso': 201,'Whippany': 202,'Oklahoma City': 203,'Melrose': 204,'Lawrence': 205,'Miami': 206,'Cleveland': 207,'Morrow': 208
    ,'Camp Lejeune': 209,'Bozeman': 210,'Winchester': 211,'Sioux Falls': 212,'Derry': 213,'Westland': 214,'Mount Horeb': 215,'East Lansing': 216,'Pensacola': 217,'Marietta': 218
    ,'Leavenworth': 219,'Fort Lauderdale': 320,'Easton': 321,'Wooster': 322,'Concord': 323,'Chula Vista': 324,'Raleigh': 325,'Westerly': 326,'Topeka': 327,'Kent': 328
    ,'Knoxville': 229,'North Hollywood': 230,'Falls Church': 231,'Hillsborough': 232,'Saint Louis': 233,'Plano': 234,'Camas': 235,'Joliet': 236,'Westport': 237,'Yakima': 238
    ,'Pacifica': 239,'Cedar Park': 240,'Springfield': 241,'Indiana': 242,'Union City': 243,'Riverside': 244,'Ada': 245,'Astoria': 246,'Alliance': 247,'Allston': 248
    ,'Scottsdale': 249,'Santa Barbara': 250,'Buckhannon': 251,'Front Royal': 252,'Frankfort': 253,'Fort Collins': 254,'Indianapolis': 255,'Sunnyvale': 256,'Manhattan Beach': 257,'Salt Lake City': 258
    ,'Torrance': 259,'San Clemente': 260,'Menifee': 261,'Carbondale': 262,'Cabin John': 263,'South Strafford': 264,'Purcellville': 265,'McKinney': 266,'Downey': 267,'Sicklerville': 268
    ,'Loganville': 269,'Hartford': 270,'Alameda': 271,'Duxbury': 272,'Schaumburg': 273,'Eden Prairie': 274,'New Braunfels': 275,'New Lenox': 276,'Corvallis': 277,'Missoula': 278
    ,'Stoughton': 279,'Pasadena': 280,'Newtonville': 281,'Eugene': 282,'Maplewood': 283,'Osseo': 284,'Lebanon': 285,'College Station': 286,'Hanover': 287,'Highland Park': 288
    ,'Huntington': 289,'Louisville': 290,'San Diego': 291,'Whitehall': 292,'Pacifica': 293,'Amherst': 294,'Newark': 295,'Grosse Pointe': 296,'Staten Island':297,'Bellingham':298,'Franklin':299
    ,'Whitewater':300,'Helena':301,'Ventura':302,'Vallejo':303,'Arcadia':304,'Broomfield':305,'Brighton':306,'Livermore':307,'Burbank':308,'Mechanicsburg':309,'Slingerlands':310,'Nogales':311
    ,'Clifton':312,'Hamilton':313,'Glen Ellyn':314,'Mankato':315,'Antrim':316,'Humble':317,'Bedford':318,'Saint Cloud':319,'Logan':320,'Worcester':321,'Seneca Falls':322,'Kalamazoo':323,'Blacksburg':324
    ,'Hagerstown':325,'Carmel':326,'Framingham':327,'Durango':328,'Orlando':329,'Kelseyville':330,'Monterey':331,'Elgin':332,'Vernon Hills':333,'Lombard':334,'Kansas City':335,'Dekalb':336
    ,'Naperville':337,'Aurora':338,'Syracuse':339,'Fayetteville':340,'-999':341,'Warrensburg':341,'Fairfax Station':342,'Watertown':343,'Lexington Park':344,'Strasburg':345,'Carlsbad':346
    ,'Emeryville':347,'San Rafael':348,'Omaha':349,'Mauldin':350,'Cockeysville':351,'Washington':352,'Young America':353,'Irvine':354,'Prescott Valley':355,'Danville':356,'Montgomery':357
    ,'Tallahassee':358,'Willington':359,'Lincoln':360,'Glens Falls':361,'Oakton':362,'Milwaukee':363,'El Cajon':364,'Marquette':365,'Decatur':366,'Burke':367,'Basye':368,'Las Cruces':369,'Bryan':370
    ,'Racine':371,'Armada':372,'Belmont':373,'Rockville':374,'Lubbock':375,'Battle Creek':376,'Richmond':377,'Waukesha':378,'Englewood':378,'Chula Vista,':379,'Austin':380,'Las Vegas':381
    ,'Oxford':381,'Long Beach':382,'Norwalk':383} 
    state = {'CA': 1,'MI': 2,'GU': 3,'SC': 4,'KS': 5,'VA': 6,'AZ': 7,'MN': 8,'NJ': 9
    ,'NY': 10,'OH': 11,'ID': 12,'WI': 13,'ME': 14,'CO': 15,'IL': 16,'MA': 17,'AR': 18
    ,'VA': 19,'NC': 20,'SC': 21,'FL': 22,'TX': 23,'MD': 24,'LA': 25,'OK': 26,'WA': 27,'OR': 28
    ,'IA': 29,'PA': 30,'CO': 31,'NH': 32,'CT': 33,'CO': 34,'TN': 35,'IN': 36,'VT': 37,'UT': 38
    ,'SD': 39,'NE': 40,'KY': 41,'RI': 42,'DE': 43,'-999':44,'GA':45,'NM':46,'HI':47,'MO':48,'PR':49,'MT':50,'DC':51,'AL':52,'NV':53,'WV':53} 
    occupations = {'academic/educator': 1,'K-12 student': 2,'clerical/admin': 3,'college/grad_student': 4
    ,'other':5,'artist': 6,'technician/engineer':7,'homemaker': 8,'writer': 9,'self-employed': 10
    ,'executive/managerial': 11,'lawyer': 12,'programmer': 13,'farmer': 14,'homemaker': 15,'sales/marketing': 16,'unemployed': 17,'scientist': 18,'lawyer':19,'doctor/health_care':20
    ,'customer service':21,'tradesman/craftsman':22,'retired':23} 


    data.gender = [gender[item] for item in data.gender] 
    data.city = [city[item] for item in data.city] 
    data.state = [state[item] for item in data.state] 
    data.occupations = [occupations[item] for item in data.occupations] 

    alldata= pd.concat([data.gender,data.age,data.city,data.state,data.latitude,data.longitude,data.occupations],1)
    labelsarray= data.ranking
    labels=labelsprocessing(labelsarray)
    dd=  alldata.values.tolist()
    myarray1 = np.array(dd)
    myarray2 = np.array(labels)
    newarray = np.concatenate((myarray1, myarray2),axis=1)
    ALL = newarray.tolist()
    # y = np.array(labels)
    # X = np.array(preprocessing.normalize(alldata, norm='l1'))  # rankdata(np.array(data))
    fmt = '%1.1f', '%1.1f', '%1.1f', '%1.1f', '%1.6f', '%1.5f', '%1.1f', '%1.1f', '%1.1f', '%1.1f', '%1.1f', '%1.1f', '%1.1f', '%1.1f'
    np.savetxt('C:\\Ayman\PhDThesis\\top7movies_ranked.txt',ALL,fmt=fmt)


loadDatapermutation('top7movies')