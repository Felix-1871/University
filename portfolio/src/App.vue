<script lang="ts">
export default {
    name: "App",
    data() {
        return {
            msg: "Welcome to my portfolio",
        };
    },
};
</script>

<template>

  
  <div class="navbar">
    <img src="./assets/felix.png" alt="">
    <h3>Felix Manning</h3>
    <p>Sometimes it takes a real man to be the best girl</p>
    <ul>
      <li><router-link to="/home">Home</router-link></li>
      <li><router-link to="/skills">Skills</router-link></li>
      <li><router-link to="/projects">Projects</router-link></li>
      <li><router-link to="/about">About me</router-link></li>
      <li><router-link to="/contact">Contact</router-link></li>
    </ul>
    <v-btn class="darkmode_toggler">
     <font-awesome-icon icon="sun"></font-awesome-icon>
    <v-tooltip
    activator="parent"
    location="bottom"
    >Toggle Dark mode</v-tooltip>
    </v-btn>
<label class="separator">|</label>
    <v-btn class="support">
     <font-awesome-icon icon="heart"></font-awesome-icon>
    <v-tooltip
    activator="parent"
    location="bottom"
    >Support me!</v-tooltip>
    </v-btn>

  </div>
<router-view></router-view>
</template>

<style scoped>
* {
  font-family: 'Poppins', sans-serif;
  font-size: 20px;
  color: #fff;
}

.navbar {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100px;
  background: #000000ea;
  display: flex;
  align-items: center;
  vertical-align: center;
  justify-content: space-between;
  padding: 0 50px;
  box-sizing: border-box;
  z-index: 999;
  border-radius: 0 0 5px 5px;
}

.navbar img {

  width: 75px;
  height: 75px;
  border-radius: 50%;
  object-fit: cover;
}

.navbar h3 {
  font-size: 30px;
  font-weight: 600;
}

.navbar p {
  font-size: 14px;
  font-weight: 300;
  padding: 0 10px;
}

.navbar ul {
  display: flex;
  list-style: none;
  /* WONDER: Wonder about whether it should be center or not */
  align-items: center;
}

.navbar ul li {
  margin: 0 20px;
}

.navbar ul li a {
  text-decoration: none;
  color: #fff;
}

.darkmode_toggler, .support {
  width: 50px;
  height: 50px;
  background: none;
  color: #fff;
  cursor: pointer;
  border: 0 1px 0 0 solid #fff;
  border-radius: 5px;
  font-size: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: 0.5s;
}


.darkmode_toggler:hover, .support:hover {
  background: #ffffff3a;
  color: #000;
}

.separator {
  color: #fff;
  font-weight: 600;
  margin: 0 10px;
}

</style>