from pymongo import MongoClient
client = MongoClient("mongodb://admin:admin@172.28.0.23:20253/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false")
mydb = client["IMAGES_livestream"]
mycol = mydb["images_new"]
db = []

print(111111, mycol.find())

for x in mycol.find({}):
    print(x)
    break

# for x in mycol.find():
#     print(x)
#     x = {'ID': x['ID'], 'ID_PRODUCT': x['ID_PRODUCT'], 'PRODUCT_NAME': x['PRODUCT_NAME'], 'IMAGE_URL': x['IMAGE_URL']}
#     db.append(x)