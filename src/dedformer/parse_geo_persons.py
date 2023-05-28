from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm.auto import tqdm
import pandas as pd


def create_driver():
    option = webdriver.ChromeOptions()
    prefs = {'profile.default_content_setting_values': {'images': 2, 'javascript': 2}}
    option.add_experimental_option('prefs', prefs)

    driver = webdriver.Chrome("C:\\chromedriver.exe", options=option)
    return driver


if __name__ == '__main__':
    users = pd.read_csv("data/users.csv")

    users['Url'] = ['https://www.google.com/maps/search/' + i for i in users['адрес проживания']]

    Url_With_Coordinates = []

    driver = create_driver()
    parsed_urls = set()
    with open('data/geocoord_persons.txt', 'r') as f:
        while line := f.readline():
            parsed_urls.add(line.split(' -> ')[0])

    for url in tqdm(set(users.Url).difference(parsed_urls), leave=False):
        while True:
            try:
                driver.get(url)
                result = driver.find_element(By.CSS_SELECTOR, 'meta[itemprop=image]').get_attribute('content')
            except:
                driver.close()
                driver = create_driver()
            else:
                break
        with open('data/geocoord_persons.txt', 'a') as f:
            f.write(f'{url} -> {result}\n')


    driver.close()
