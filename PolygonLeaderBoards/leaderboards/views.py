from django.shortcuts import render, redirect
from django.contrib.auth.models import auth, User
from django.contrib import messages
from .models import LeaderBoards


def index(request):
    return render(request, "html/home.html")


def leader_boards(request):
    lbs = LeaderBoards.objects.filter(specification="Lung_CT")
    lbs = lbs.values_list(
        'user_name', 'model_name', 'accuracy', 'precision', 'recall', 'f1_score'
    )
    lbs = dict([(str(num + 1), value) for num, value in enumerate(lbs)])

    return render(request, "html/learderboards.html", {'lbs': lbs})


def upload(request):
    return render(request, "html/upload.html")


def results(request):
    return render(request, "html/results.html")


def publish_results(request):
    publishment = LeaderBoards(
        specification="Lung_CT",
        user_id=request.user.id,
        user_name=request.user.username,
        model_name="ResNet",
        accuracy=0.88,
        precision=0.871,
        recall=0.871,
        f1_score=0.871
    )
    publishment.save()
    return leader_boards(request)


def delete_publishment(request, specification, user_id):
    publishment = LeaderBoards.objects.get(specification=specification, user_id=user_id)
    publishment.delete()
    return userpage(request)


def userpage(request):
    user_results = LeaderBoards.objects.filter(user_id=request.user.id)
    user_lbs = user_results.values_list(
        'specification', 'model_name', 'accuracy', 'precision', 'recall', 'f1_score'
    )
    user_lbs = dict([(value[0], value[1:]) for value in user_lbs])

    return render(request, "html/userpage.html", {'user_lbs': user_lbs})


def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return index(request)
        else:
            messages.info(request, 'Invalid Username or Password')
            return render(request, "html/login.html")

    return render(request, "html/login.html")


def signup(request):
    if request.method == "POST":
        firstname = request.POST.get('firstname')
        lastname = request.POST.get('lastname')
        email = request.POST.get('email')
        username = request.POST.get('username')
        password = request.POST.get('password')

        if User.objects.filter(username=username).exists():
            messages.info(request, 'User with such Username is already exist')
            return render(request, "html/signup.html")
        elif User.objects.filter(email=email).exists():
            messages.info(request, 'User with such Email is already exist')
            return render(request, "html/signup.html")
        else:
            user = User.objects.create_user(
                username=username,
                password=password,
                email=email,
                first_name=firstname,
                last_name=lastname
            )
            user.set_password(password)
            user.save()
            return login(request)

    return render(request, "html/signup.html")


def logout(request):
    auth.logout(request)
    return index(request)
