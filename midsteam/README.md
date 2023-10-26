Nuxt WebApp Documentation
=========================

Overview
--------

This documentation provides a comprehensive guide for setting up and understanding a Nuxt web application that uses the CheapShark API. The web application allows users to browse and purchase games from various online retailers at discounted prices. Project made as a course credit for Database Apps course.

Table of Contents
-----------------

1.  [Introduction](https://chat.openai.com/#introduction)
2.  [Setup](https://chat.openai.com/#setup)
    -   [Prerequisites](https://chat.openai.com/#prerequisites)
    -   [Installation](https://chat.openai.com/#installation)
3.  [Configuration](https://chat.openai.com/#configuration)
    -   [CheapShark](https://chat.openai.com/#cheapshark)
    -   [Supabase](https://chat.openai.com/#supabase)
4.  [Usage](https://chat.openai.com/#usage)
    -   [Development](https://chat.openai.com/#development)
    -   [Build](https://chat.openai.com/#build)
    -   [Deployment](https://chat.openai.com/#deployment)
5.  [Features](https://chat.openai.com/#features)
    -   [Game Listing](https://chat.openai.com/#game-listing)
    -   [Game Details](https://chat.openai.com/#game-details)
    -   ["Purchases"](https://chat.openai.com/#"purchases")
6.  [Extending Functionality](https://chat.openai.com/#extending-functionality)
    -   [Additional API Endpoints](https://chat.openai.com/#additional-api-endpoints)
    -   [User Authentication](https://chat.openai.com/#user-authentication)
7.  [Troubleshooting](https://chat.openai.com/#troubleshooting)
8.  [Contributing](https://chat.openai.com/#contributing)
9.  [License](https://chat.openai.com/#license)

Introduction
------------

The Nuxt WebApp is a client-side rendered web application built using the Nuxt.js framework, which is based on Vue.js. It utilizes the CheapShark API to retrieve game data and facilitate game purchases from multiple online retailers, and Supabase as its database.

The web application provides features such as game listing, game details, and purchase functionality. Users can browse games, view their details, and make purchases by redirecting to the respective retailer's website.

Setup
-----

### Prerequisites

Before setting up the Nuxt WebApp, ensure that you have the following prerequisites installed:

1.  Node.js (v12 or higher)
2.  npm (Node Package Manager)
3.  Git

### Installation

To install and set up the Nuxt WebApp, follow these steps:

1.  Clone the project repository from GitHub:

    shell

-   `git clone https://github.com/Felix-1871/nuxt-webapp.git`

    -   Change into the project directory:

    shell

    -   `cd nuxt-webapp`

    -   Install the project dependencies using npm:

    shell

1.  `npm install`

Configuration
-------------

Before running the Nuxt WebApp, you need to configure the Supabase API Key and url.

### Cheapshark

To obtain the CheapShark API, follow these steps:

1.  Visit the [CheapShark API](https://apidocs.cheapshark.com/) documentation website.
2. Read what you need to
3. Copy it.

### Supabase

1. Create Database
2. Create Tables
3. Copy and paste URL and API Key

Usage
-----

### Development

To run the Nuxt WebApp in development mode, use the following command:

shell

`npm run dev`

This command starts the development server and makes the web application accessible at `http://localhost:3000`. Any changes you make to the source code will trigger hot-reloading and update the application in real-time.

### Build

To build the Nuxt WebApp for production, execute the following command:

shell

`npm run build`

This command compiles and optimizes the source code, generating a production-ready version of the web application in the `dist` directory.

### Deployment

To deploy the Nuxt WebApp to a production environment, you can follow the deployment instructions provided by your hosting platform. The generated `dist` directory contains the static files that can be served by a web server.

Features
--------

The Nuxt WebApp offers the following key features:

### Game Listing

The web application displays a list of games retrieved from the CheapShark API. The games are presented in a paginated manner, allowing users to navigate through multiple pages. Each game item displays its title, price, and thumbnail image.

### "Purchases"

The web application provides a "purchase" functionality by showing the user alert that he made purchase. When the user decides to purchase a game, the query is sent to Supabase table.

Extending Functionality
-----------------------

The Nuxt WebApp can be extended in various ways to add more functionality based on your requirements. Here are a couple of examples:

### Additional API Endpoints

To integrate additional API endpoints or expand the functionalities related to game purchases, you can modify the relevant components and pages. You can fetch data from other APIs, handle different responses, and implement the necessary logic to support new features.

### User Authentication

If you want to add user authentication to the web application, you can leverage Nuxt.js authentication modules such as [Nuxt Auth](https://auth.nuxtjs.org/) or [Nuxt Firebase](https://firebase.nuxtjs.org/). These modules provide features like user registration, login, and authentication flows, allowing you to secure parts of the application or personalize the user experience.

Troubleshooting
---------------

If you encounter any issues or have questions while setting up or using the Nuxt WebApp, consider the following troubleshooting steps:

1.  Ensure that all the prerequisites are installed correctly.
2.  Verify that the API key and endpoint for the CheapShark API are correctly configured in the project.
3.  Check the console output for any error messages or warnings when running the application.
4.  Search for existing GitHub issues or relevant community forums for similar problems.

If the issue persists, consider opening a new issue on the project's GitHub repository, providing detailed information about the problem and steps to reproduce it.

Contributing
------------

If you would like to contribute to the Nuxt WebApp project, you can follow these steps:

1.  Fork the project repository.
2.  Create a new branch for your contribution.
3.  Implement your changes or new features.
4.  Commit and push your changes to your forked repository.
5.  Open a pull request to the original project repository, describing your changes and the motivation behind them.

License
-------

The Nuxt WebApp project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You can use, modify, and distribute the codebase in accordance with the terms and conditions specified by the license.
