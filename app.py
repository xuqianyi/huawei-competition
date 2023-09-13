import re
import random
from numpy import load
from mysql_operator import *
from datetime import datetime as dt
from flask import Flask, request, jsonify, render_template,redirect, url_for

ab_list = ['1AVB', '2AVB', '3AVB', 'ALMI', 'AMI', 'ANEUR', 'ASMI', 'CLBBB',
        'CRBBB', 'DIG', 'EL', 'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS',
        'INJIL', 'INJIN', 'INJLA', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL',
        'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD',
        'LAFB', 'LAO/LAE', 'LMI', 'LNGQT', 'LPFB', 'LVH', 'NDT', 'NORM',
        'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'WPW']

app = Flask(__name__)
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/test", methods=['POST', 'GET'])
def test():
    if request.method == 'POST':
        num_question = int(request.form["numQuestions"])
        return redirect(url_for("quiz", num_question = num_question))
    else:
        return render_template('test.html')

@app.route('/report/<favorites>/<mistakes>/<score>/', methods=['POST', 'GET'])
def report(favorites, mistakes, score):
    return render_template('report.html', favorites = favorites, mistakes = mistakes, score = score)

@app.route('/quiz/<num_question>', methods=['POST', 'GET'])
def quiz(num_question):
    if request.method == 'POST':
        print(request.form["favorites"])
        print(request.form["wrong"])
        print(request.form["score"])
        return redirect(url_for("report", favorites = request.form["favorites"], mistakes = request.form["wrong"], score = request.form["score"]))
    else:
        num_question = int(num_question)
        questions = randomly_select_data(num_question)
        if len(questions) < num_question:
            return jsonify({"code": -1, "message": "Don't have enough data", "data":{}})
        returnQuestions = []
        for i in range(len(questions)):
            random_numbers = random.sample(range(0, 44), 4)
            answer = questions[i][2] - 1
            while answer in random_numbers:
                random_numbers = random.sample(range(0, 44), 4)
            correct_answer = random.randint(0, 3)
            random_numbers[correct_answer] = answer
            returnQuestions.append({
                "id": questions[i][0],
                "pictureUrl": questions[i][1],
                "options": [ab_list[random_numbers[0]], ab_list[random_numbers[1]], ab_list[random_numbers[2]], ab_list[random_numbers[3]]],
                "correct_answer": correct_answer
            })
        return render_template('quiz.html', returnQuestions = returnQuestions)

@app.route("/plan_main", methods=['POST', 'GET'])
def plan_main():
    if request.method == 'POST':
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]
        days = (dt.strptime(end_date, "%Y-%m-%d") - dt.strptime(start_date, "%Y-%m-%d")).days
        num_qn = int(request.form["numQuestions"])
        # insert_plan(num_qn, start_date, end_date, days)
        return redirect(url_for("my_plan", plan_days=days))
    else:
        return render_template('plan_main.html')

@app.route("/my_plan/<plan_days>", methods=['POST', 'GET'])
def my_plan(plan_days):
    return render_template('my_plan.html', days=plan_days)

@app.route("/learn", methods=['POST', 'GET'])
def learn():
    return render_template('learn.html')

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
    app.run(host = '0.0.0.0', debug = True)