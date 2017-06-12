import pymongo, re, datetime
import numpy as np
from operator import itemgetter

client = pymongo.MongoClient('mongodb://ubicomp:ubicomp2017!@smart-iot.kaist.ac.kr:27017/data')
db = client.get_default_database()
collections = db.collection_names(include_system_collections=False)

# O is study, 1 is social
STUDY = 0
SOCIAL = 1
WINDOW_SIZE = 5    # Minute
WINDOW_LEN = 60 * WINDOW_SIZE   # CUT dataX by unit of WINDOW

observation = [ [1496383380000, 10, {'M21':STUDY, 'M25':STUDY, 'M22':STUDY}],   # 6/2 15:03
                [1496557950000, 20, {'M21':STUDY, 'M25':STUDY, 'M26':STUDY}],# 6/4 15:32
                [1496661360000, 18, {'M21':STUDY, 'M22':SOCIAL, 'M11':STUDY, 'M26':SOCIAL}],# 6/5 20:16
                [1496754300000, 16, {'M11':STUDY, 'M25':STUDY, 'M12':STUDY,
                                     'M23':SOCIAL}], # 6/6 22:05
                [1496811240000, 20, {'M12':SOCIAL, 'M22':SOCIAL}], # 6/7 13:54
                [1496812740000, 40, {'M15':SOCIAL}], # 6/7 14:19
                [1496818440000, 20, {'M21':STUDY, 'M22':STUDY}], # 6/7 15:54
                [1496983140000, 14, {'M12':SOCIAL, 'M25':STUDY, 'M15':STUDY}], # 6/9 13:39
                [1496988540000, 14, {'M11':STUDY, 'M26':SOCIAL, 'M15':STUDY, 'M13':STUDY}], # 6/9 15:09
                [1496989440000, 14, {'M11':STUDY, 'M22':STUDY, 'M15':STUDY, 'M13':STUDY}]    # 6/9 15:24
                ]

sorted(observation, key=itemgetter(0))

def create_dataset():
    raw_dataX = []
    raw_dataY = []

    for collection in collections:
        if collection != 'N1TwosomePlace_data':
            continue

        col = db[collection]
        doc = col.find(
            filter={"$and": [
                {"name": re.compile('^M1|^M2')},
                {"timestamp": {
                    "$gte":1496361600000}}
            ]},
            projection={'name': True,
                        'value': True,
                        'timestamp': True}
        )

        if doc is not None:
            len_observation = len(observation)
            obs_index = 0

            sensors = list(observation[obs_index][2].keys())
            temp_value = dict.fromkeys(list(observation[obs_index][2].keys()))

            for sensor in sensors:
                temp_value[sensor] = [[observation[obs_index][0], 0.5]]

            for d in doc:
                leftBound = observation[obs_index][0]
                rightBound = leftBound + observation[obs_index][1] * 60000

                if leftBound > d['timestamp']:
                    continue

                if d['name'] in sensors:
                    if d['timestamp'] < rightBound:
                        temp_value[d['name']].append([d['timestamp'], 1 if d['value'] else 0])
                else:
                    continue

                if d['timestamp'] > rightBound:
                    for sensor in sensors:
                        temp_value[sensor].append([rightBound, 0.5])

                    # SAVE temp_value to raw_dataX, raw_dataY
                    for i in range(len(sensors)):
                        if observation[obs_index][2][sensors[i]] == STUDY:
                            raw_dataX.append(temp_value[sensors[i]])
                            raw_dataY.append([1, 0])
                        else:
                            raw_dataX.append(temp_value[sensors[i]])
                            raw_dataY.append([0, 1])

                    obs_index += 1

                    if (obs_index == len_observation):
                        break

                    sensors = list(observation[obs_index][2].keys())
                    temp_value = dict.fromkeys(list(observation[obs_index][2].keys()))
                    for sensor in sensors:
                        temp_value[sensor] = [[observation[obs_index][0], 0.5]]

                    continue

                # newFormat = datetime.datetime.fromtimestamp(d['timestamp'] / 1000.0)
                # f.write("%s,%s,%s,%s,%s,%s,%d\n" %
                #         (d['name'], newFormat.month, newFormat.day, newFormat.hour, newFormat.minute, newFormat.second, d['value']))

    # print (raw_dataX)
    return raw_dataX, raw_dataY


def refine_dataset(raw_dataX, raw_dataY, WINDOW_LEN):
    # ADD padding 0.5 to raw_dataX
    dataX = []

    for dataset in raw_dataX:
        temp = []
        for i in range(len(dataset)):
            if i == len(dataset)-1:
                break

            t1 = dataset[i][0] // 1000
            t2 = dataset[i+1][0] // 1000

            t_gap = t2 - t1
            num_pad = t_gap - 1

            if t_gap == 0:  # 0, 1 were delivered at the same time
                dataset[i+1][0] = dataset[i+1][0] + 1000

            temp.append(dataset[i][1])
            if num_pad >= 1:
                temp += num_pad * [0.5] # For padding EMPTY time-series
            # else:
            #     print(t1, t2)

        dataX.append(np.asarray(temp))
    # print (len(dataX))

    # WINDOW SHIFT
    train_X = []
    train_Y = []

    for k in range(len(dataX)):
        dataset = dataX[k]
        for i in range(len(dataset) - WINDOW_LEN - 1):
            a = dataset[i:(i + WINDOW_LEN)]
            train_X.append(a)
            train_Y.append(raw_dataY[k])

    return np.asarray(train_X), np.asarray(train_Y)


def load_dataset(WINDOW_LEN):
    rawX, rawY = create_dataset()
    train_X, train_Y = refine_dataset(rawX, rawY, WINDOW_LEN)

    return (train_X, train_Y)


def class_selection(trainPredict, testPredict):
    resultTrain = []
    resultTest = []

    for predict in trainPredict:
        if predict[0] > predict[1]:
            resultTrain.append([1, 0])
        else:
            resultTrain.append([0, 1])

    for predict in testPredict:
        if predict[0] > predict[1]:
            resultTest.append([1, 0])
        else:
            resultTest.append([0, 1])

    return np.asarray(resultTrain), np.asarray(resultTest)


if __name__ == '__main__':
    rawX, rawY = create_dataset()
    train_X, train_Y = refine_dataset(rawX, rawY, WINDOW_LEN)

    print(train_X.shape)
    print(train_Y.shape)

