from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = '#$#$#$#$#$#$'
app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql://root:%s@localhost/ecommerce'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)





@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)