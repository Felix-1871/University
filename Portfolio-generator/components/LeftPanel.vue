<template>
 <div class="panel left">
    <div class="avatar-wrapper">
        <img src="/99283735.png" alt="avatar">
    </div>
    <p class="desc nick">{{FormData.nick}}</p>
    <p class="desc"> <a :href="FormData.links_linkedin">LinkedIn</a></p>
    <p class="desc"> <a :href="FormData.links_github">Github</a></p>
    
    
 </div>
</template>

<style scoped>

.panel {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    background-color: #1e1e2e;
}

.left {
    width: 200px;
}

.avatar-wrapper {
    width:140px; 
    height:140px;
    margin: 0 auto;
    border-radius: 50%;
    overflow: hidden;
    border: 1px solid #28283E;
}

.avatar-wrapper img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.desc {
    margin: 0;
    padding: 0;
    font-weight: 300;
    font-family: 'Montserrat', sans-serif;
    color: #cdd6f4;
}

.nick {
    font-size: 20px;
}

.desc a{
    color: #cdd6f4;
    text-decoration: none;
}



</style>

<script>

export default {
    name: 'LeftPanel',
    props: {
        FormData: Object,
    },
}

</script>