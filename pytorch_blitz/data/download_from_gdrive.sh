nk_id="10bOQTnjbb-npZO6KdanTCtwtU0E4kRcu"
model_name="cifar-10-python.tar.gz"

URL="https://docs.google.com/uc?export=download&id=$model_link_id"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$model_link_id" -O $model_name && rm -rf /tmp/cookies.txt

model_link_id="15NulSVQTmtFgzc0IEcBxcusM4VtTyF5P"
model_name="faces.zip"

URL="https://docs.google.com/uc?export=download&id=$model_link_id"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$model_link_id" -O $model_name && rm -rf /tmp/cookies.txt

