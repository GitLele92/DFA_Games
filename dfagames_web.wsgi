activate_this = '/home/fuggitti/virtualenvs/dfagames/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

#!/usr/bin/python3
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/dfagames_web")

from ltlf2dfa_web import app as application