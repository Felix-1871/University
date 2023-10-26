<template>
<div class="FormContent" v-if="!form_submitted">
   
    <h1>Hi! Enter your data here and it will show up on your portfolio!</h1>
 <form @submit.prevent="submitForm">
    <h2></h2>
 <label for="nick">Name:</label>
      <input type="text" id="nick" v-model="FormData.nick">
      <h2> Links to other sites</h2>
      <label for="links_github">Github:</label>
      <input type="text" id="links_github" v-model="FormData.links_github" >
      <label for="links_linkedin">LinkedIn:</label>
      <input type="text" id="links_linkedin" v-model="FormData.links_linkedin" >
      <h2> What you're gonna write here, will be on your about me page </h2>
      <label for="about_me">About me:</label>
      <input type="text" id="about_me" v-model="FormData.about_me" >
      <h2> What you're gonna write here, will be on your skills page </h2>
      <label for="skills_proficient">Skills i'm proficient at:</label>
      <input type="text" id="skills_proficient" v-model="FormData.skills" >
      <label for="skills_meh">Skills I'm beginning to master: </label>
      <input type="text" id="skills_meh" v-model="FormData.skills_meh" >
      <h2> What you're gonna write here, will be on your projects page </h2>
      <label for="projects">Projects:</label>
      <input type="text" id="projects" v-model="FormData.projects" >
      <h2> Links to contact you</h2>
      <label for="contact_email">Email:</label>
      <input type="text" id="contact_email" v-model="FormData.contact_email" >
      <label for="contact_discord">Discord:</label>
      <input type="text" id="contact_discord" v-model="FormData.contact_discord" >
      <button type="submit">Submit</button>
    </form>
</div>
<div class="interface" v-else>
    <MainInterface :FormData="FormData"></MainInterface>
</div>
</template>

<script>
import MainInterface from "./MainInterface.vue";


export default {
    name: "Form",
    data() {
        return {
            FormData: {
                nick: "Felix",
                links_github: "https://github.com/Felix-1871",
                links_linkedin: "https://linkedin.com/in/test-1-37b9859b",
                about_me: "It's my personal website!",
                skills: "Linux, C#, Rust",
                skills_meh: "Java, Python, Ruby",
                projects: "Currently I'm working on a new Linux Distro, more info soon :)",
                contact_email: "test@test.com",
                contact_discord: "https://discordapp.com/users/940588929562841108"
            },
            form_submitted: false
        };
    },
    methods: {
        submitForm(e) {
            e.preventDefault();
            this.form_submitted = true;
            console.log(this.FormData);
        }
    },
    components: { MainInterface}
}
</script>

<style scoped>

.FormContent {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    max-height: 100vh;
    color: var(--ctp-mocha-text);
    overflow: auto;
    padding-bottom: 50px;
    
}

.FormContent h1 {
    font-size: 3rem;
    margin-bottom: 2rem;
    justify-content: center;
    text-align: center;
}

.FormContent form {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 50%;
}

.FormContent form label {
    font-size: 1rem;
    margin-bottom: .5rem;
}

.FormContent form input {
    width: 100%;
    height: 2rem;
    margin-bottom: 1rem;
    border-radius: 5px;
    border: 1px solid var(--ctp-mocha-text);
    background-color: var(--ctp-mocha-background);
    color: var(--ctp-mocha-text);
}

.FormContent form button {
    width: 100%;
    height: 2rem;
    margin-top: 1rem;
    border-radius: 5px;
    border: 1px solid var(--ctp-mocha-text);
    background-color: var(--ctp-mocha-background);
    color: var(--ctp-mocha-text);
    font-size: 1rem;
}

.FormContent form button:hover {
    background-color: var(--ctp-frappe-base);
    color: var(--ctp-frappe-text);
    transform: scale(1.1);
    transition: 0.5s;
}
.FormContent {
    animation: slideInFromLeft 1s ease-out;
}

@keyframes slideInFromLeft {
    0% {
        transform: translateX(-100%);
    }
    100% {
        transform: translateX(0);
    }
}





.FormContent form button:focus {
    outline: none;
}

.FormContent form input:focus {
    outline: none;
}




</style>