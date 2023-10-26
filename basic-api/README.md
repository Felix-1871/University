# Basic API Project

## Overview
This project is a basic API built using Python and Flask for university class. It provides a simple interface for interacting with a database of quotes about AI. Readme was generated using ChatGPT.

## Installation
1. Clone the repository

  `git clone https://github.com/Felix-1871/basic-api`

  2. Install the required packages

`pip install -r requirements.txt`

 3. Run the application

`python server.py`

## Usage

The API should have the following endpoints (only get was succesfully executed):

   - GET /quotes
        Retrieves a list of all items in the database
   - GET /quotes/<id>
        Retrieves a specific item by its id
  -  POST /quotes
        Creates a new item in the database
        The request body should be in the following format:
        
<code>{
    "id": "id",
        "author": "author-name",
        "quote": "quote"
  }</code>
   - DELETE /quotes/<id>
        Deletes an existing item by its id
