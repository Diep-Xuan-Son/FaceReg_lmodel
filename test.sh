#!/bin/bash
curl -X 'POST' 'http://192.168.6.161:8421/api/searchUserv2' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'image=@tb2.jpeg;type=image/jpeg'