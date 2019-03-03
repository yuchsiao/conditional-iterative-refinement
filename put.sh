rsync -arvh \
    --exclude __pycache__ \
    --exclude data/glove.840B.300d \
    --exclude data/glove.840B.300d.zip \
    --exclude emb/* \
    --exclude emb.tar.gz \
    * mr85p01ad-gpu017:~/project/squad
