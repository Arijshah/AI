import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/physical-ai/introduction-to-physical-ai/what-is-physical-ai">
            Start Learning Physical AI 🤖
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Learning Embodied Intelligence Through Hands-On Lessons">
      <HomepageHeader />
      <main>
        <div className="container padding-vert--xl">
          <div className="row">
            <div className="col col--4">
              <div className="card padding--md margin--sm">
                <div className="text--center">
                  <h3>🤖 Embodied Intelligence</h3>
                  <p>Learn how AI interacts with the physical world through sensors and actuators.</p>
                </div>
              </div>
            </div>
            <div className="col col--4">
              <div className="card padding--md margin--sm">
                <div className="text--center">
                  <h3>🧠 Robot Brains</h3>
                  <p>Dive into ROS 2, NVIDIA Isaac, and the software stacks that power modern robots.</p>
                </div>
              </div>
            </div>
            <div className="col col--4">
              <div className="card padding--md margin--sm">
                <div className="text--center">
                  <h3>🌐 Digital Twins</h3>
                  <p>Simulate and train your robots in virtual environments before deploying to the real world.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}
