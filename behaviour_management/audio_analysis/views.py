from django.shortcuts import render
from django.http import JsonResponse, Http404
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from audio_analysis.audio_ai_model.audio_class import audio_analysis
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
import os

# Create your views here.


# @csrf_exempt
class api(APIView):
    permission_classes = (IsAuthenticated,)
    def get(self, request):
        content = {'message': 'GET Response'}
        return Response(content)

    permission_classes = (IsAuthenticated,)    
    def post(self,request):
        user_id=request.POST['user id']
        test_id=request.POST['test id']
        lang=request.POST['lang']
        weight=eval(request.POST['weights'])

        if request.FILES['audio_file']:
            file=request.FILES['audio_file']
            fs=FileSystemStorage()

            user_id_list=os.listdir('media/audio_dumps/')
            if user_id in user_id_list:
                pass
            else:
                os.mkdir('media/audio_dumps/'+user_id)
            audio_path="media/audio_dumps/"+user_id+"/"+test_id+".wav"
            file_name=fs.save(audio_path,file)
            print("google api hit-----------")
            obj1=audio_analysis(audio_path)
            # import pdb; pdb.set_trace()
            if lang == 'English':
                # print(file_name)
                print(audio_path)
                obj1.audio_process(audio_path,weight)
            elif lang == 'Hindi':
            	obj1.audio_process(audio_path,weight,'HI')

            try:
                print(obj1.sentiment_score)
            except:
                obj1.sentiment_score={}

            content={'user id':user_id,'test id':test_id,'final_score':obj1.final_score,'text_nlp_score':obj1.sentiment_score,'deceptive score':obj1.deceptive_score}

            return JsonResponse(content)
        # raise Http404("URL NOT FOUND")
