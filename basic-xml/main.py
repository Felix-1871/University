import xml.etree.ElementTree as ET
from lxml import etree
import time

# parse the XML file
tree = etree.parse('data.xml')
root = tree.getroot()
firstName = root.xpath('//firstName')
lastName = root.xpath('//lastName')
age = root.xpath('//age')
debtAmount = root.xpath('//debtAmount')
debtDate = root.xpath('//debtCollectionDate')

def _create():
    # create new elements
    new_person = etree.Element("person")

    firstName = etree.SubElement(new_person, "firstName")
    firstName.text = input("Value for the field Name: ")

    lastName = etree.SubElement(new_person, "lastName")
    lastName.text = input("Value for the field Surname: ")

    age = etree.SubElement(new_person, "age")
    age.text = input("Value for the field Age: ")

    debtAmount = etree.SubElement(new_person, "debtAmount")
    debtAmount.text = input("Value for the field Debt Amount: ")

    debtDate = etree.SubElement(new_person, "debtCollectionDate")
    debtDate.text = input("Value for the field Debt Collection Date: ")

    # add new element to the root element
    root.append(new_person)

    # write the modified XML to the file
    tree.write("data.xml")
    print('Correctly created element')
    time.sleep(3)
    start()
def _read():

   blabla = input('What value do you want to read? ')
   try:
    number = input('From which position do you want to read value? ')      


    match blabla:
        case 'firstName':
         print(firstName[int(number) -1].text) 
         time.sleep(1)
         _read()
        case 'lastName':
            print(lastName[int(number) -1].text) 
            time.sleep(1)
            _read()
        case 'age':
            print(age[int(number) -1].text) 
            time.sleep(1)
            _read()
        case 'debtAmount':
           print(debtAmount[int(number) -1].text)
           time.sleep(1)
           _read()
        case 'debtDate':
         print(debtDate[int(number) -1].text)
         time.sleep(1)
         _read()
        case 'all':
         print(firstName[int(number) -1].text) 
         print(lastName[int(number) -1].text) 
         print(age[int(number) -1].text) 
         print(debtAmount[int(number) -1].text)
         print(debtDate[int(number) -1].text)
         time.sleep(1)
         _read()
        case _:
            print('Unknown value')
            time.sleep(3)
            start()
   except:
    print("that position doesn't exist. Try different one")
    _read()
def _update():
    
        element_name = input('Which element do you want to update? ')
        if (element_name == "firstName" or element_name == "lastName" or element_name == "age" or element_name == "debtAmount" or element_name == "debtDate"):
            number = input('Which position do you want to update? ')
            number = int(number) -1
           # concatenating every input to create selection for xpath
            xpath_expression = '//person' + '[' + str(number) + ']/' + element_name

            selectedElement = root.xpath(xpath_expression)
            selectedElement[0].text = input('Enter new value: ')
           # writing to file
            tree.write("data.xml")
            print("Succesfully updated selected element")
            time.sleep(3)
            start()
        else:
            print("Unknown element")
            time.sleep(3)
            _update()

def _delete():

   number = input('Which row do you want to delete? ')
   xpath_expression = '//person[' + number +']' 
   person = root.xpath(xpath_expression)
   print( 'You want to delete' + ' ' + firstName[int(number) -1].text + ' ' + lastName[int(number) -1].text + ' aged ' + age[int(number) -1].text + ' with debt being collected on ' + debtDate[int(number) -1].text + ' which amounts to ' + debtAmount[int(number) -1].text )
   sure = input('Are you sure? (y/n) ')
   
   if(sure == 'y'):
    person[0].getparent().remove(person[0])
    tree.write("data.xml")
    print('Delete action complete')
    time.sleep(3)
    start()
   else:
    print('Delete action cancelled')
    time.sleep(3)
    _delete()


def start():
    action = input('What action do you want to take? ')
    match action:
        case 'update':
            _update()
        case 'create':
            _create()
        case 'read':
            _read()
        case 'delete':
            _delete()
        case _:
            print('Unknown command')
            time.sleep(3)
            start()

start()
