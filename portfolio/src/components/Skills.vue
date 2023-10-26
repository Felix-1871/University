<script setup lang="ts">
import { Html5Icon, LinuxmintIcon, CsharpIcon, PythonIcon, GithubIcon, VuedotjsIcon, VisualstudiocodeIcon, ArchlinuxIcon, WindowsIcon, GitIcon, Css3Icon
 } from "vue3-simple-icons";
import { useSkillsStore } from "../stores/SkillsStore";
const skills = useSkillsStore();


</script>

<template>
        <h1>Skills</h1>
  
    <!-- Loop through the skills data from SkillsStore and create buttons dynamically -->
    <div class="container">
      <button v-for="skill in skills" :key="skill.id" :skill="skill" @click="dialog = true">
        <v-icon :size="50" :icon="skill.icon" />

      </button>


    </div>

    <!-- <v-dialog v-model="dialog" max-width="500">
      <v-card>
        <v-card-title>
          Overlay Title
        </v-card-title>
        <v-card-text>
          
          <v-container>
          </v-container>
        </v-card-text>
        <v-card-actions>
          <v-btn color="primary" @click="dialog = false">
            Close
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog> -->
</template>

<style scoped>
* {
  font-family: 'Poppins', sans-serif;
  font-size: 20px;
  color: #fff;
}
.container {  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  grid-template-rows: 1fr 1fr 1fr;
  gap: 10px 10px;
  grid-auto-flow: row;
  justify-content: center;
  align-content: center;
  justify-items: center;
  align-items: center;
  grid-template-areas:
    ". . ."
    ". . ."
    ". . ."
    ". . .";
}
.container button{
  background: transparent !important; 
    border: 1px rgba(255, 255, 255, 0.103) solid;
    border-radius: 15px;
    width: 100px;
    height: 100px;
    box-shadow: 
    12px 12px 16px 0 rgba(0, 0, 0, 0.25),
    -8px -8px 12px 0 rgba(255, 255, 255, 0.3);
    margin: 10px;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
}
.container button:hover{
    background-color: rgba(255, 255, 255, 0.4);
    transform: scale(1.1);
}

</style>