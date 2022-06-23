from ntpath import join
from unittest import result
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model


def app():
    st.title("Veriler")

    data = pd.read_csv("./onceki_lig_siralamalari.csv")
    pred_data = pd.read_csv("./play_off_takimlari_lig_siralamasi.csv")

    lig_yillari = list(range(2003, 2022))

    result = st.sidebar.selectbox("Yıl Seçin", lig_yillari)
    analiz = st.sidebar.selectbox(
        "Istatistikler", ("Veriler", "Korelasyon Matrisi", "Veri Analizleri")
    )
    # algoritmalar = st.sidebar.selectbox("Model Eğitimi", ("Veriler", "Güncel Veriler", "Linear Regression",
    #    "Random Forest", "XGBoost"))

    if analiz == "Veriler":
        yil_secimi = data[data["Year"] == result]
        st.write(yil_secimi)

    data = data.drop(columns=["Rk", "O_MP", "L", "PW", "PL", "Arena"]).rename(
        columns={"eFG%.1": "O_eFG%", "TOV%.1": "O_TOV%", "FT/FGA.1": "O_FT/FGA"}
    )
    pred_data = pred_data.drop(columns=["Rk", "O_MP", "L", "PW", "PL", "Arena"]).rename(
        columns={"O_eFG%_1": "O_eFG%", "O_TOV%_2": "O_TOV%"}
    )

    corr = data.corr().abs()
    corr = corr.loc[corr["Playoff Wins"] > 0.25]

    variables = list(corr.index)
    corr_df = data[variables].groupby("Playoff Wins").mean()

    if analiz == "Korelasyon Matrisi":
        fig1 = plt.figure(figsize=(12, 8))
        plt.title(
            "Play-Off Maç Galibiyetlerine Dayanarak Her Katagorideki Ortalama Lig Sırası"
        )
        frs = sns.heatmap(corr_df, annot=True)
        frs.set(xlabel="Lig Sıralaması")
        frs.set(ylabel="Play-Off Galibiyet Sıralaması")
        st.write("2003-2021 Arası Verilerin Korelasyon Matrisi")
        st.write(fig1)
        loooong_text = " ".join(
            [
                "Sezon Maçlarını Kazanma (W), Margin of Victory (MOV), Simple Rating System (SRS) verilerinde ilk 3 sıralamada olmak Şampiyonluğu belirlemede önemli kıstaslardan olduğu görünüyor. SRS ve MOV birbirlerine çok yakın olduklarından modellemede SRS'i kullanmaya karar verdik. Çünkü SRS verileri sezon takviminin zorluğunu ve takım oyuncularının +/- istatistiklerini de içerdiğinden daha doğru sonuçlar alınmasında etkili. MOV maçın kaç farkla kazanıldığının verileridir."
            ]
        )
        st.text(
            """Filtrelendikten sonra elimizde kalan veriler:
                1.   FG% (Field Goal %)
                2.   3P% (3 Point %)
                3.   2P% (2 Point %)
                4.   O_FG% (Opponent Field Goal %)
                5.   O_2P% (Opponent 2 Point %)
                6.   O_DRB (Opponent Defensive Rebounds)
                7.   O_BLK (Opponent Blocks)
                8.   W (Regular Season Wins)
                9.   MOV (Margin of Victory)
                10.  SRS (Simple Rating System)
                11.  ORtg (Overall Offensive Rating)
                12.  DRtg (Defensive Rating)
                13.  eFG% (Effective Field Goal %)
                14.  O_eFG% (Opponent Effective Field Goal %)
                15.  Attendance\n"""
        )
        st.markdown(loooong_text)

        fig3 = plt.figure(figsize=(12, 8))

        plt.title("Correlation between Filtered Variables")
        fltr = sns.heatmap(data=data[variables].corr(), annot=True)
        st.pyplot(fig3)

        ############### VERİ ANALİZLERİ ###################
    if analiz == "Veri Analizleri":
        fig2, ax = plt.subplots()
        frs = sns.regplot(x=data["2P%"], y=data["Playoff Wins"], label="2P%")
        sns.regplot(x=data["FG%"], y=data["Playoff Wins"], label="FG")
        sns.regplot(x=data["eFG%"], y=data["Playoff Wins"], label="eFG%")
        frs.set(xlabel="Lig Sıralaması")
        frs.set(ylabel="Playoff Galibiyetleri")
        plt.legend()
        plt.title(
            "Playoff Galibiyetleri ve Normal Sezon Şut Verimliliği Arasındaki Korelasyon"
        )
        st.write(fig2)
        long_text = "".join(
            [
                "Effective Field Goal Yüzdeleri (eFG% ve O_eFG%) karşılık geldikleri\nalt nitelikler olan (2P%, O_2P%, FG%, O_FG%) yüksek korelasyona sahip. Bu yüzden bu 4 veriyi çıkartıp, şut yüzdelerinin doğruluğunun biraz daha fazla olduğu EFG değerlerini kullanacağız."
            ]
        )
        st.markdown(long_text)
        st.text(
            """Model eğitimi için karar kıldığımız veriler şunlar:
                1.   3P%
                2.   Opponent blocks
                3.   Regular Season Wins
                4.   Simple Rating System
                5.   Overall Offensive Rating
                6.   Overall Defensive Rating
                7.   Effective Field Goal %
                8.   Opponent Effective Field Goal %\n"""
        )

        ###### Regression

        fig4 = plt.figure(figsize=(12, 8))
        frs2 = sns.regplot(x=data["2P%"], y=data["Playoff Wins"], label="2P%")
        frs3 = sns.regplot(x=data["FG%"], y=data["Playoff Wins"], label="FG")
        frs4 = sns.regplot(x=data["eFG%"], y=data["Playoff Wins"], label="eFG%")
        frs2.set(xlabel="Lig Sıralaması")
        frs2.set(ylabel="Playoff Galibiyetleri")
        plt.legend()
        plt.title(
            "Playoff Galibiyetleri ve Normal Sezon Şut Verimliliği Arasındaki Korelasyon"
        )
        st.pyplot(fig4)

        ###### Distributions

        frs5 = sns.displot(
            x=data["O_BLK"], y=data["Playoff Wins"], kind="kde", bw_adjust=1, fill=True
        )
        plt.title(
            "Normal Sezon Karşılaşmalarında Yapılan Maç Başı Blokların Lig Sıralaması Dağılımı"
        )
        st.pyplot(frs5)
        st.markdown(
            "O_BLK istatistiğinin, Playoff galibiyetlerine yaptığı etki yüksek. O_BLK sıralaması yüksek olan takımları savunması daha zor oluyor. Bu da kazanma olasılığını arttırıyor."
        )

        frs6 = plt.figure(figsize=(10, 5))
        plt.title("eFG% ve O_BLK arasındaki ilişki")
        frs = sns.regplot(x=data["O_BLK"], y=data["eFG%"])
        frs.set(xlabel="O_BLK Lig Sıralaması", ylabel="eFG% Lig Sıralaması")
        st.pyplot(frs6)

        st.markdown(
            "Play-offlarda 16 galibiyete ulasan takimlar sampiyon olur. 2003'ten bu yana sampiyon olan takimlarin listesi."
        )
        sampiyonlar = data.loc[data["Playoff Wins"] == 16]
        yillar = sampiyonlar.Year
        veri = sampiyonlar
        sampiyonlar

        ##### HÜCUM

        st.subheader("Hücum")

        frs7 = plt.figure(figsize=(13, 5))
        plt.xticks(np.arange(min(yillar), max(yillar) + 1, 1.0))
        plt.title("Şampiyon Takımların Yıllara Gore Hücum İstatistik Dağılımları")
        frs = sns.lineplot(x=yillar, y=veri["3P%"], label="3P%")
        frs = sns.lineplot(x=yillar, y=veri["eFG%"], label="eFG%")
        frs = sns.lineplot(x=yillar, y=veri["O_BLK"], label="O_BLK")
        frs = sns.lineplot(x=yillar, y=veri["ORtg"], label="ORtg")
        plt.ylim(0, 30)
        frs.set(xlabel="Yıllar")
        frs.set(ylabel="Normal Sezon Lig Sıralamaları")
        st.pyplot(frs7)
        st.markdown(
            "2000li yıllardan 2010lu yıllara kadar şampiyon takımların normal sezon hücum anlayışlarındaki çeşitlilik çok fazla. Hem ofansif hem de defansif olarak takımlar her alana gereken önemi vermiş. 2014 yılından itibaren Golden State Warriors (GSW) takımı çok fazla 3'lük atmaya başlayarak basketbolda devrim yapmıştır. Bu yıldan itibaren oyun içerisinde 3 sayılık atış diğer sayı bulma yöntemlerinden çok daha fazla önem taşımıştır. Genellikle son yıllarda şampiyon olan takımların 3 sayılık yüzdesi sıralamasında ilk 5'te olduğunu görüyoruz. Sadece 2020 yılında koronavirüsten dolayı verilen ara nedeniyle, aynı zamanda GSW takımının bu sene Play-Offlara şutörlerinin sakatlığı ve formsuzluğu nedeniyle katılamamasıyla 3lük yüzdelerinde düşüklük görüyoruz. Bu sene şampiyon olan takım daha çok savunmasıyla ön plana çıkmış."
        )

        ##### SAVUNMA
        st.subheader("Savunma")
        frs8 = plt.figure(figsize=(13, 5))
        plt.xticks(np.arange(min(yillar), max(yillar) + 1, 1.0))
        plt.title("Şampiyon Takımların Yıllara Gore Defans İstatistik Dağılımları")
        frs = sns.lineplot(x=yillar, y=veri["O_eFG%"], label="O_eFG%")
        frs = sns.lineplot(x=yillar, y=veri["DRB"], label="DRB")
        frs = sns.lineplot(x=yillar, y=veri["DRtg"], label="DRtg")
        plt.ylim(0, 30)
        frs.set(xlabel="Yıllar")
        frs.set(ylabel="Normal Sezon Lig Sıralamaları")
        st.pyplot(frs8)
        st.markdown(
            "Normal sezon takım savunma verileri, hücum verilerinden daha az çeşitlilik gösteriyor. Bu verilerde ilk 5 ya da 10'da olmak şampiyonunun kim olacağına dair iyi bir işaret denebilir. 2019'a kadar efektif şut yüzdesi ve genel defans sıralamasının önemi defansif ribaund sıralamasından yüksek olsa da, bu yıldan sonra şampiyon olan takımların defansif ribaund sıralamasında ligde ne kadar iyilerse o kadar şampiyon olabileceğinin göstergesi denilebilir. Takımlar savunmaya daha fazla önem vermeye başlamış ve rakip takımlarının efektif şut yüzdeleri sıralamaları bu seneden sonra düşmeye başlamış."
        )

        ###### Simple Rating System (SRS) ile Maç Kazanma ilişkisi
        st.subheader("Simple Rating System (SRS) ile Maç Kazanma ilişkisi")
        frs9 = plt.figure(figsize=(13, 5))
        plt.xticks(np.arange(min(yillar), max(yillar) + 1, 1.0))
        plt.title("Şampiyon Takımların Yıllara Göre Kazanma İstatistik Dağılımları")
        frs = sns.lineplot(x=yillar, y=veri["W"], label="W")
        frs = sns.lineplot(x=yillar, y=veri["SRS"], label="SRS")
        plt.ylim(0, 30)
        frs.set(xlabel="Yıllar")
        frs.set(ylabel="Normal Sezon Lig Sıralamaları")
        st.pyplot(frs9)
        st.markdown(
            "Simple Rating System (SRS) ve takımların maç kazanma sayıları yıllara göre her zaman istikrarlı bir şekilde devam etmiş. SRS sıralamasında normal sezon maçlarında ilk 5'in dışarısında olan bir takımın şampiyon olması çok nadir olmuş. SRS, oyuncuların maç içerisinde gösterdikleri performanslara göre hesaplanır. Bu performanslar Lig Sıralamalarını belirleyen verilere dayanır. Oyuncular iyi performans gösterirlerse istatistik verilerinde +, kötü performans gösterirlerse - puan alırlar. Bu da maça yaptıkları katkıyı belirler. Katkısı yüksek olan oyuncunun sakatlanması ya da sezon ortasında farklı bir takıma takas olması takımların Lig Sıralamasında etkili olur."
        )

        ##### Hücum Genel Sıralamasının (ORtg), Defansif Genel Sıralamasının(DRtg), Maç Galibiyet Sıralamasının(W) Karşılaştırması
        st.subheader(
            "Hücum Genel Sıralamasının (ORtg), Defansif Genel Sıralamasının(DRtg), Maç Galibiyet Sıralamasının(W) Karşılaştırması"
        )
        frs10 = plt.figure(figsize=(13, 5))
        plt.xticks(np.arange(min(yillar), max(yillar) + 1, 1.0))
        plt.title(
            "Şampiyon Takımların Yıllara Göre Hücum-Defans-Galibiyet Sıralanmalarının İstatistiği"
        )
        frs = sns.lineplot(x=yillar, y=veri["W"], label="W")
        frs = sns.lineplot(x=yillar, y=veri["DRtg"], label="DRtg")
        frs = sns.lineplot(x=yillar, y=veri["ORtg"], label="ORtg")
        plt.ylim(0, 30)
        frs.set(xlabel="Yıllar")
        frs.set(ylabel="Normal Sezon Lig Sıralamaları")
        st.pyplot(frs10)
        st.markdown(
            "2000lerden bu yana Hücum Genel Sıralamalası ile Defansif Genel Sıralamaları arasında büyük farklarla birlikte Defansif Genel Sıralaması her zaman önemli olmuşken son yıllarda Hücum Genel Sıralaması etkili olmaya başlamış."
        )
