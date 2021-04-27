----------------------------------------------------------------------
-- DESCRIPTION: Upload data from S3 to Redshift
-- AUTHOR: Tyler Wayne
-- LAST MODIFIED: 2019-06-21
----------------------------------------------------------------------

copy {table}
from '{bucket}/{filename}'
access_key_id '{aws_access_key_id}'
secret_access_key '{aws_secret_access_key}'
delimiter as '{delimiter}'
gzip
null as '';
