import pymongo, re, datetime

client = pymongo.MongoClient('mongodb://ubicomp:ubicomp2017!@smart-iot.kaist.ac.kr:27017/data')
db = client.get_default_database()
collections = db.collection_names(include_system_collections=False)

def _find_sensor_data():
    f = open("./result.csv", "w")
    for collection in collections:
        col = db[collection]
        doc = col.find(
            filter={"$and": [
                {"name": re.compile('^M1|^M2')},
                {"timestamp": {
                    "$gte":1496361600000,
                    "$lt": 1496707200000}}
            ]},
            projection={'name': True,
                        'value': True,
                        'timestamp': True}
        )
        if doc is not None:
            for d in doc:
                flag = 0
                if d['value'] == 'TRUE':
                    flag = 1

                newFormat = datetime.datetime.fromtimestamp(d['timestamp'] / 1000.0)
                f.write("%s,%s,%s,%s,%s,%s,%d\n" %
                        (d['name'], newFormat.month, newFormat.day, newFormat.hour, newFormat.minute, newFormat.second, flag))


if __name__ == '__main__':
    _find_sensor_data()
