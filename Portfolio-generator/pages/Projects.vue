<template>
<div class="ProjectsContent">
    <h1>Projects</h1>
    <p>Here are my projects</p>
    <div class="projects">
    {{ FormData?.projects }}</div>

</div>
</template>

<script lang="ts">

export default {
    name: 'Projects',
    props: {
        FormData: Object,
    },
}

</script>

<style scoped>

.ProjectsContent h1 {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-align: center;

}

.ProjectsContent p {
    font-size: 1.5rem;
    font-weight: 400;
    margin-bottom: 1rem;
    text-align: center;
}

.ProjectsContent .projects {
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 1rem;
    text-align: center;
    text-decoration: none;
    color: var(--ctp-mocha-text);
}



</style>