from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

file = open('songs.txt', 'w')

driver = webdriver.Chrome(CHROMEDRIVER_PATH)

songs = ['notafraid', 'beautiful', 'loseyourself', 'rapgod', 'killshot', 'godzilla', 'mockingbird', 'whenimgone', 'venommusicfromthemotionpicture', 'kamikaze', 'singforthemoment', 'tillicollapse', 'neverloveagain', 'youreneverover', 'fall', 'spacebound', 'intoodeep', 'stepdad', 'marsh', 'onfire', 'littleengine']

songs_sraped = 0
total_songs = len(songs)

for song in songs:
    time.sleep(5)
    driver.get("https://www.azlyrics.com/lyrics/eminem/{}.html".format(song))

    element = driver.find_element_by_xpath("/html/body/div[3]/div/div[2]/div[5]")

    file.write(element.text)

    print("scraped {}".format(song))
    songs_sraped+=1
    print("{} scraped out of {} songs".format(songs_sraped, total_songs))


