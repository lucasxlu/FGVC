"""
Label the number annotations with Baidu AI, the groundtruth label is obtained by randomly selecting 11 images belong
to one category, and take the majority voting strategy to decide the final annotation
"""
import os
import json
import sys


def get_access_token():
    import requests

    try:
        baidu_ai_cfg = json.load(open('./baidu_ai_cfg.json'))
    except:
        print('Please provide baidu_ai_cfg.json config file firstly...')
        sys.exit(0)

    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={0}&client_secret' \
           '={1}'.format(baidu_ai_cfg['AK'], baidu_ai_cfg['SK'])
    response = requests.post(host, headers={'Content-Type': 'application/json; charset=UTF-8'})
    content = response.json()

    if 'error' not in content.keys():
        return content['access_token']
    else:
        return 'ERROR'


def rec_with_baidu_ai(img_file, access_token):
    """
    Plant Recognition with Baidu AI
    :param img_file:
    :param access_token:
    :return:
    """
    import base64
    import requests

    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/plant"

    f = open(img_file, 'rb')
    img = base64.b64encode(f.read())

    params = {"image": img}

    request_url = request_url + "?access_token=" + access_token
    response = requests.post(url=request_url, data=params, headers={'Content-Type':
                                                                        'application/x-www-form-urlencoded'})
    result = response.json()
    if result:
        return result['result']


if __name__ == '__main__':
    access_token = get_access_token()

    # img_dir = "G:\Dataset\CV\FGCV5/train\category_697"
    img_dir = "C:\DataSet\FGVC/train\category_782"
    for _ in os.listdir(img_dir):
        print(rec_with_baidu_ai(os.path.join(img_dir, _), access_token))
