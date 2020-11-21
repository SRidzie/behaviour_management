import speech_recognition as sr
from googletrans import Translator
import imp
import sys
sys.modules["sqlite"] = imp.new_module("sqlite")
sys.modules["sqlite3.dbapi2"] = imp.new_module("sqlite.dbapi2")
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from pydub import AudioSegment
import time
import multiprocessing
import tensorflow as tf
import librosa
import numpy as np
import subprocess
#ys.modules["bz2"] = imp.new_module("bz2")
#import pandas as pd
import csv



nltk.download('vader_lexicon') #one time activity.




class audio_analysis:
    def __init__(self,audio_path):
        self.audio_path=audio_path


    # def extract_audio_from_video(self,video_path=None,audio_path=None):
    #     self.video_path=video_path
    #     if video_path != None:
    #
    #         self.audio_path=audio_path
    #         os.system("ffmpeg -i "+video_path+" "+self.audio_path)



    def __text_extract(self,audio_file,lang=None):
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
            if lang==None:
                try:
                    self.text=r.recognize_google(audio)
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
            elif lang=='HI':
                try:
                    self.text=r.recognize_google(audio,language='hi-IN')
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))




    def audio_cutter(self):
        clip = AudioSegment.from_mp3(self.audio_path)
        self.clip_names=[]
        clip_name_count=0
        start_sec=0
        total_duration=int(clip.duration_seconds)
        count=total_duration//30+1



        while count>0:
            try:
                # audio_name=self.audio_path.split('/')[-1]
                audio_name=self.audio_path.split('.')[0]

                extract=clip[start_sec:(start_sec+30000)]
                file_name=audio_name+"_extract_"+str(clip_name_count)+'.wav'
                #video store in extract_data folder
                extract.export(file_name, format="wav")
                start_sec+=30000
                clip_name_count+=1
                count-=1
                self.clip_names.append(file_name)
            except:
                break



    def __trans_to_eng(self):
        t=Translator()
        self.r=t.translate(self.string, dest='en')
        self.string=self.r.text



    def sentiment_analysis(self,weight):


        sid=SentimentIntensityAnalyzer()
        if self.string != "":
            self.sentiment_score=sid.polarity_scores(self.string)
            score = self.sentiment_score['neu']*weight['neu']+self.sentiment_score['pos']*weight['pos']-self.sentiment_score['neg']*weight['neg']+self.sentiment_score['compound']*weight['compound']
            self.sentiment_score['score']=score*100
            self.sentiment_score['text']=self.string
        else:
            self.sentiment_score="NOT able to find words in audio"


    def get_text(self,lang=None):
        """ lang = EN or HI """
        self.string=""
        for a in self.clip_names:
            if lang == None:
                self.__text_extract(a,lang)
                time.sleep(1)
                self.string+=self.text
            elif lang =="HI":

                self.__text_extract(a,lang)
                time.sleep(1)
                self.string+=self.text
        if lang == None:
            self.__trans_to_eng()



    def feature_extraction(self):
        #audio_path='24 Jan, 12.48 PM.mp3'
        if self.audio_path != None:
            try:
                test_data, sr = librosa.load(self.audio_path)
            except:
                return False
            else:
                df1=[]
                for a in range(0,len(test_data),550):
                    mfccs = np.mean(librosa.feature.mfcc(y=test_data[a:a+(550)], sr=sr, n_mfcc=40).T,axis=0)
                    df1.append(mfccs)
                self.test=np.array(df1).reshape(-1,40,1)
        else:
            self.test=None


    def prediction(self):
        if self.test.shape[0]!=0 or self.test != None:
            model=tf.keras.models.load_model("audio_analysis/audio_ai_model/speech_deceptive_model.h5")
            pred=model.predict_classes(self.test)
            pred=pred.tolist()
            score=(pred.count(1)/len(pred))*100
            self.deceptive_score={'score':score,'raw data':{'truth':pred.count(1),'deceptive':pred.count(0),'total':len(pred)}}


        else:
            self.deceptive_score = 'No result from prediction'

    def score(self):
        self.final_score=self.sentiment_score['score']*0.5+self.deceptive_score['score']*0.5

    def __store_result_to_csv(self):
        with open('audio_result.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file,fieldnames=['audio_name','text_nlp_score','deceptive_speech_score'])
            writer.writerow({'audio_name':self.audio_path,'text_nlp_score':self.sentiment_score,'deceptive_speech_score':self.deceptive_score})

    def remove_audio(self):
        for a in self.clip_names:
            try:
                os.remove(a)
            except:
                pass

    def fun1(self,lang):
        self.get_text(lang)
        self.sentiment_analysis()

    def get_text_multiprocess(self,lang):
        p = multiprocessing.Process(target=self.fun1,name='fun1' ,args=(lang,))
        p.start()

        # Wait 10 seconds for foo
        time.sleep(120)

        # Terminate foo
        p.terminate()

        # Cleanup
        p.join()



    #
    #
    # def timer_process(self,lang):
    #     lock = mp.Lock()
    #     pm=mp.Process(target=self.fun1, args=(lang))
    #     pm.start()
    #     import psutil
    #     p = psutil.Process(pm.pid)
    #     try:
    #         p.wait(timeout=60)
    #     except psutil.TimeoutExpired:
    #         p.kill()
    #         raise




    def audio_process(self,audio_path,weight,lang=None):
        # import pdb; pdb.set_trace()
        # self.extract_audio_from_video(video_path,audio_path)
        self.audio_cutter()
        #self.get_text_multiprocess(lang)
        # self.timer_process(lang)
        self.get_text(lang)
        self.sentiment_analysis(weight)
        # print(self.string)
        self.feature_extraction()
        self.prediction()
        self.score()
        # self.__store_result_to_csv()
        self.remove_audio()


    # def __audio_process(self,l,video_path,audio_path,lang):
    #     l.acquire()
    #     import pdb; pdb.set_trace()
    #     self.extract_audio_from_video(video_path,audio_path)
    #     self.audio_cutter()
    #     self.get_text(lang)
    #     self.sentiment_analysis()
    #     self.feature_extraction()
    #     self.prediction()
    #     # self.__store_result_to_csv()
    #     self.remove_audio()
    #
    #
    #     l.release()
    #
    #
    #
    #
    #
    # def audio_multiprocess(self,video_path,audio_path,lang=None):
    #     lock = mp.Lock()
    #     p=mp.Process(target=self.__audio_process, args=(lock,video_path,audio_path,lang))
    #     p.start()
    #     p.join()
