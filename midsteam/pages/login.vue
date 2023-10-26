<template>
    <div class="LoginForm">
        <h1>Login</h1>
        <form @submit.prevent="login">
            <label for="email">Email</label>
            <input type="email" name="email" v-model="email" />
            <label for="password">Password</label>
            <input type="password" name="password" v-model="password" />
            <button type="submit">Login</button>
        </form>
        <p>Don't have an account? <a href="/register">Register</a></p>


    </div>
</template>

<style scoped>
.LoginForm {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    width: 100%;
    background-color: var(--ctp-mocha-crust);
}

.LoginForm h1 {
    color: var(--ctp-mocha-text);
}

.LoginForm form {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 50%;
}

.LoginForm form input {
    width: 100%;
    margin: 10px 0;
    padding: 10px;
    border: 2px solid #fff;
    border-radius: 10px;
    background-color: var(--ctp-mocha-mantle);
    color: var(--ctp-mocha-text);
}

.LoginForm form button {
    width: 100%;
    margin: 10px 0;
    padding: 10px;
    border: 2px solid #fff;
    border-radius: 10px;
    background-color: var(--ctp-mocha-mantle);
    color: var(--ctp-mocha-text);
}

.LoginForm p {
    color: var(--ctp-mocha-text);
}

.LoginForm p a {
    color: var(--ctp-mocha-text);
}
</style>

<script setup lang="ts">
import { createClient } from '@supabase/supabase-js'
import dotenv from 'dotenv'

dotenv.config()

const supabaseUrl = "https://ttgymjmisrdktsonzgoa.supabase.co"
const supabaseKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR0Z3ltam1pc3Jka3Rzb256Z29hIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODYwODMxMDAsImV4cCI6MjAwMTY1OTEwMH0.AjFgT0mwVjlJm8H_lcajqZTXU7C72epYg4uLpYenosY"
const supabase = createClient(supabaseUrl, supabaseKey)

async function login() {
  // Get the user's email and password from the form
const email = ""
// Define a password data property
const password = ref('')

// Bind the password data property to the password input field using v-model


  // Sign in the user with Supabase
  const { user, error } = await supabase.auth.signIn({
    email,
    password,
  })

  if (error) {
    // Handle the error
  } else {
    // Redirect the user to the dashboard
  }
}

</script>