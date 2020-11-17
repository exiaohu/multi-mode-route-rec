import json

from datetime import datetime
from flask import Flask, jsonify, request
from shapely.geometry import Point, Polygon

app = Flask(__name__)

default_res = {
    'plans': [
        {
            'summary': {
                'descriptions': "途径：学院路 > 四环 > 北湖渠路",
                'costs': {
                    'distance': 13.0,
                    'time': 22,
                    'price': 22.0,
                    'transfer_time': 0,
                }
            },
            'path': [
                {
                    'type': 'bus',
                    'path': {
                        "coordinates": [
                            [116.34924069326846, 39.93912181555644],
                            [116.33897282483227, 39.965277426042334],
                            [116.3338043267136, 39.97515081171022],
                            [116.33158933111197, 39.99157237586751],
                            [116.31407490649673, 40.031716819809645],
                            [116.30868937119958, 40.039809159854634],
                            [116.30020270159802, 40.05176697606433],
                            [116.31330779662255, 40.06959112466942],
                            [116.32995722135033, 40.0694775923388],
                            [116.35407407457015, 40.070490516946336],
                            [116.40608847994501, 40.05162199849808],
                            [116.42827787555503, 40.04160203389973],
                            [116.44367982820653, 39.99435394529868],
                            [116.42968443491644, 39.976245639366674],
                            [116.42552641548052, 39.96694236069532],
                            [116.42649607565718, 39.95676377950603],
                            [116.42961691720929, 39.940236985677544]
                        ]
                    }
                }
            ]
        },
        {
            'summary': {
                'descriptions': "途径：西土城路 > 三环 > 京承高速",
                'costs': {
                    'distance': 17.4,
                    'time': 23,
                    'price': 27.0,
                    'transfer_time': 3,
                }
            },
            'path': [
                {
                    'type': 'bus',
                    'path': {
                        "coordinates": [
                            [116.609772885186104, 40.051608746591413],
                            [116.587001294037648, 40.078261971791562],
                            [116.450815058924675, 39.960153052596297],
                            [116.429616917209287, 39.940236985677544]
                        ]
                    }
                }
            ]
        },
        {
            'summary': {
                'descriptions': "途径：四环 > 北辰东路 > 科荟路",
                'costs': {
                    'distance': 13.5,
                    'time': 24,
                    'price': 23.0,
                    'transfer_time': 5,
                }
            },
            'path': [
                {
                    'type': 'bus',
                    'path': {
                        "coordinates": [
                            [116.422144675270616, 39.844456779879081],
                            [116.442176865028841, 39.832851791547633],
                            [116.453067064200951, 39.82660729452369],
                            [116.454637104286718, 39.805571223582653],
                            [116.474218011298689, 39.801721012573516],
                            [116.484579879343428, 39.805628894409359],
                            [116.499407270113565, 39.80175410813851],
                            [116.507356575218338, 39.791996671065299],
                            [116.5157559767771, 39.781665255049901],
                            [116.53403261326784, 39.771846093792419],
                            [116.556511228112768, 39.782605047745513],
                            [116.57557650663901, 39.794075514172114],
                            [116.585728200740419, 39.802464451567836],
                            [116.596139600866678, 39.811573756732713]
                        ]
                    }
                }
            ]
        }
    ]
}

MAX_LNG, MIN_LNG = 116.495, 116.265
MAX_LAT, MIN_LAT = 39.995, 39.820

vld_area = Polygon(((MAX_LNG, MAX_LAT), (MAX_LNG, MIN_LAT), (MIN_LNG, MIN_LAT), (MIN_LNG, MAX_LAT), (MAX_LNG, MAX_LAT)))


def error(msg='Some error.'):
    return jsonify({'success': False, 'message': msg})


def ok(data=None):
    return jsonify({'success': True, 'data': data})


@app.route('/api/routing')
def routing():
    args = request.args

    origin = Point(json.loads(args['origin_location'])['lng'], json.loads(args['origin_location'])['lat'])
    dest = Point(json.loads(args['dest_location'])['lng'], json.loads(args['dest_location'])['lat'])

    assert vld_area.contains(origin) and vld_area.contains(dest)

    modals = args.getlist('modals') or ['walking', 'driving', 'taxi', 'public']

    assert len({'walking', 'driving', 'taxi', 'public'}.union(modals)) == 4

    timestamp = datetime.fromtimestamp(int(args['timestamp'])) if 'timestamp' in args.keys() else datetime.now()

    pref = args.get('preference', 'default')

    assert pref in ('default', 'distance', 'time', 'price', 'transfer_time')

    total = args.get('total', default=3, type=int)

    print(*locals().values(), sep='\n')

    return ok(default_res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='2500')
