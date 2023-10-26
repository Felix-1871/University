<template>
    <div class="RegisterForm">
        <h1>Register</h1>
        <form @submit.prevent="register">
            <div class="form-group">
                <label for="username">Username</label>
                <input v-model="username" type="text" name="username" id="username" placeholder="Username" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input v-model="password" type="password" name="password" id="password" placeholder="Password" required>
            </div>
            <div class="form-group">
                <label for="password2">Confirm Password</label>
                <input v-model="password2" type="password" name="password2" id="password2" placeholder="Confirm Password" required>
            </div>
            <button type="submit">Register</button>
        </form>
    </div>
</template>

<script>
import { createClient } from '@supabase/supabase-js'


const supabaseUrl = 'https://ttgymjmisrdktsonzgoa.supabase.co'
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR0Z3ltam1pc3Jka3Rzb256Z29hIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODYwODMxMDAsImV4cCI6MjAwMTY1OTEwMH0.AjFgT0mwVjlJm8H_lcajqZTXU7C72epYg4uLpYenosY'
const supabase = createClient(supabaseUrl, supabaseKey)


export default {
    data() {
        return {
            username: '',
            password: '',
            password2: ''
        }
    },
    methods: {
     async register() {
        if (this.password !== this.password2) {
            alert('Passwords do not match!')
            return
        }
        else{

const { data, error } = await supabase
  .from('users')
  .insert([
    { username: this.username, password: this.password },
  ])
  if (error) {
    console.log(error)
    } else {
    alert('User created successfully!')
}


        }
    },
}
}


</script>

<style scoped>

.RegisterForm {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    width: 100%;
    background-color: var(--ctp-mocha-crust);
}

.RegisterForm h1 {
    color: var(--ctp-mocha-text);
}

.RegisterForm form {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 50%;
}

.RegisterForm form input {
    width: 100%;
    margin: 10px 0;
    padding: 10px;
    border: 2px solid #fff;
    border-radius: 10px;
    background-color: var(--ctp-mocha-mantle);
    color: var(--ctp-mocha-text);
}

.RegisterForm form button {
    width: 100%;
    margin: 10px 0;
    padding: 10px;
    border: 2px solid #fff;
    border-radius: 10px;
    background-color: var(--ctp-mocha-mantle);
    color: var(--ctp-mocha-text);
}

.RegisterForm form button:hover {
    background-color: var(--ctp-mocha-crust);
    color: var(--ctp-mocha-text);
}

.RegisterForm form label {
    color: var(--ctp-mocha-text);
}

.RegisterForm form a {
    text-decoration: none;
    color: var(--ctp-mocha-blue);
}

.RegisterForm form a:hover {
    color: var(--ctp-mocha-text);
}



</style>