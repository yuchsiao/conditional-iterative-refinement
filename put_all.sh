rsync -arvh \
    --exclude __pycache__ \
    --exclude data/glove.840B.300d \
    --exclude data/glove.840B.300d.zip \
    --exclude emb/* \
    --exclude emb.tar.gz \
    --exclude save \
    * mr85p01ad-gpu017:~/project/squad

rsync -arvh \
    --exclude __pycache__ \
    --exclude data/glove.840B.300d \
    --exclude data/glove.840B.300d.zip \
    --exclude emb/* \
    --exclude emb.tar.gz \
    --exclude save \
    * mr85s01ad-stggpu001:~/project/squad

rsync -arvh \
    --exclude __pycache__ \
    --exclude data/glove.840B.300d \
    --exclude data/glove.840B.300d.zip \
    --exclude emb/* \
    --exclude emb.tar.gz \
    --exclude save \
    * mr85s01ad-stggpu002:~/project/squad

rsync -arvh \
    --exclude __pycache__ \
    --exclude data/glove.840B.300d \
    --exclude data/glove.840B.300d.zip \
    --exclude emb/* \
    --exclude emb.tar.gz \
    --exclude save \
    * mr85s01ad-stggpu003:~/project/squad

rsync -arvh \
    --exclude __pycache__ \
    --exclude data/glove.840B.300d \
    --exclude data/glove.840B.300d.zip \
    --exclude emb/* \
    --exclude emb.tar.gz \
    --exclude save \
    * mr85s01ad-stggpu004:~/project/squad
