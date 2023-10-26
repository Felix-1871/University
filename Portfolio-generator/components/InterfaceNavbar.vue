
import { NuxtLink } from '~/.nuxt/components';


import { NuxtLink } from '~/.nuxt/components';

<template>
    <div class="navbar">
        <nav class="tabs">
            <ul>
                <li><NuxtLink to="/">About</NuxtLink></li>
                <li><NuxtLink to="/skills">Skills</NuxtLink></li>
                <li><NuxtLink to="/projects">Projects</NuxtLink></li>
                <li><NuxtLink to="/resume">Resume</NuxtLink></li>
                <li><NuxtLink to="/contact">Contact</NuxtLink></li>
            </ul>
        </nav>
    </div>
</template>

<style scoped>

.navbar {
    display: flex;
    flex-direction: row;
    justify-content: flex-end;
    align-items: center;
    background-color: #1e1e2e;
}

.navbar ul {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    list-style-type: none;
    margin: 0 20px;
}

.navbar ul li {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    margin: 0 20px;
}

.navbar ul li a {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    text-decoration: none;
    color: #cdd6f4;
    font-size: 20px;
    font-weight: 100;
    font-family: 'Montserrat', sans-serif;
}

.navbar ul li a:hover {
    color: #cdd6f4;
    text-decoration: underline;
}








</style>