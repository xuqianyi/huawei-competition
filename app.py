from flask import Flask, request, jsonify, render_template
from mysql_operator import *
from numpy import load
import re

app = Flask(__name__)
@app.route("/")
def home():
    return render_template('test.html')

# @app.route("/test", methods=['GET'])
# def test():
#     return render_template('test.html')

# 选择题目，返回题目
@app.route('/start_test', methods=['POST', 'GET'])
def start_testing():  # put application's code here
    num_question = int(request.form["numQuestions"])
    questions = randomly_select_data(num_question)
    if len(questions) < num_question:
        return jsonify({"code": -1, "message": "Don't have enough data", "data":{}})
    return_dict = {"code": 0, "message": "Success", "data": []}
    for i in range(len(questions)):
        question_path = "/Users/qianyi/projects/huawei-competition/back-end/npy_files/row_" + str(questions[i][0]) + ".npy"
        data = load(question_path)
        return_dict["data"].append({"id": questions[i][0],
                                    "points": list(data),
                                    "correct_answer": questions[i][1]})
    return jsonify(return_dict)


# 上传错题
@app.route('/upload_mistakes', methods=['POST'])
def upload_mistakes():
    questions = re.findall("(-?[0-9]\d*)", request.form["mistakes"])
    uid = request.form["uid"]
    status_code, message = insert_mistakes(uid, questions)
    return jsonify({"code": status_code, "message": message})


# 上传favorite
@app.route('/upload_favorites', methods=['POST'])
def upload_favorites():
    questions = re.findall("(-?[0-9]\d*)", request.form["favorites"])
    uid = request.form["uid"]
    status_code, message = insert_favorite(uid, questions)
    return jsonify({"code": status_code, "message": message})


# 返回所有错题
@app.route('/get_mistakes', methods=['POST'])
def get_mistakes():
    uid = request.form["uid"]
    questions = select_mistakes(uid)
    return_dict = {"code": 0, "message": "Success", "data": []}
    for i in range(len(questions)):
        question_path = "npy_files/row_" + str(questions[i][0]) + ".npy"
        data = load(question_path)
        return_dict["data"].append({"id": questions[i][0],
                                    "points": list(data),
                                    "correct_answer": questions[i][1]})
    return jsonify(return_dict)

# 返回所有favorite
@app.route('/get_favorites', methods=['POST'])
def get_favorites():
    uid = request.form["uid"]
    questions = select_favorites(uid)
    return_dict = {"code": 0, "message": "Success", "data": []}
    for i in range(len(questions)):
        question_path = "/Users/qianyi/projects/huawei-competition/back-end/npy_files/row_" + str(questions[i][0]) + ".npy"
        data = load(question_path)
        return_dict["data"].append({"id": questions[i][0],
                                    "points": list(data),
                                    "correct_answer": questions[i][1]})
    return jsonify(return_dict)

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)