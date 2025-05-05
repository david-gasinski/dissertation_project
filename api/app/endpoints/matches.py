from flask import Blueprint, jsonify
from app import limiter

matches = Blueprint(
    "matches",
    __name__, 
    url_prefix="/matches",
)


@matches.route('/get', methods=['GET', 'POST'])
@limiter.limit('3/second')
def get():
    return jsonify({
        "success": True,
        "payload": {
            "message" : "LOL"
        }
    })
    
