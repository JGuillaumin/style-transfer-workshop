#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests, re
import os, time, sys
import json
import hashlib


def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        f = open(filename, 'wb')
        f.write(response.content)
        f.close()
        print("OK", filename)
    else:
        print("Error downloading file", url)


if __name__ == "__main__":

    folder = sys.argv[1]
    style = sys.argv[2]

    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(os.path.join(folder, style)):
        os.makedirs(os.path.join(folder, style))

    page = 1
    while 0 < page:
        # all
        if folder == "featured":
            url = "https://www.wikiart.org/en/paintings-by-style/%s?select=featured&json=2&page=%d" % (style, page)
        else:
            url = "https://www.wikiart.org/en/paintings-by-style/%s?json=2&page=%d" % (style, page)
        # featured
        response = requests.get(url)
        if response.status_code == 200:
            dict_files = response.json()
            print("Page %d, %d paintings total" % (page, dict_files["AllPaintingsCount"]))

            if dict_files["Paintings"] is None:
                page = 0
            else:
                for p in dict_files["Paintings"]:
                    p["year"] = str(p["year"])
                    p["width"] = str(p["width"])
                    p["height"] = str(p["height"])
                    filename = str(p["paintingUrl"]).replace("/", "_")[4:] + ".jpeg"
                    download_file(p["image"], os.path.join(folder, style, filename))
                page += 1
        else:
            print("Error", response.status_code, url)
