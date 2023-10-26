<template>

<div class="SkillsContent">
    <h1>Skills</h1>
    <p>I know these well</p>
    <p class="skills">
    {{ FormData?.skills }}</p>
 <p>I'm still learning these</p>
<p class="skills">
    {{ FormData?.skills_meh }}</p>
</div>

</template>

<script lang="ts">

export default {
    name: 'Skills',
    props: {
        FormData: Object,
    },
}

</script>

<style scoped>

.SkillsContent h1 {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-align: center;

}

.SkillsContent p {
    font-size: 1.5rem;
    font-weight: 400;
    margin-bottom: 1rem;
    text-align: center;
}

.SkillsContent .skills {
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 1rem;
    text-align: center;
    text-decoration: none;
    color: var(--ctp-mocha-text);
}


</style>