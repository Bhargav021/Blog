import { pathToRoot } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { i18n } from "../i18n"

const PageTitle: QuartzComponent = ({ fileData, cfg, displayClass }: QuartzComponentProps) => {
  const title = cfg?.pageTitle ?? i18n(cfg.locale).propertyDefaults.title
  const baseDir = pathToRoot(fileData.slug!)
  const avatarSrc = `${baseDir}/assets/profile-pic.jpg`
  return (
    <h2 class={classNames(displayClass, "page-title")}>
      <a href={baseDir} class="page-title-inner">
        <img src={avatarSrc} alt={`${title} profile photo`} class="page-title-avatar" />
        <span>{title}</span>
      </a>
    </h2>
  )
}

PageTitle.css = `
.page-title {
  font-size: 1.75rem;
  margin: 0;
  font-family: var(--titleFont);
}

.page-title-inner {
  align-items: center;
  display: inline-flex;
  gap: 0.55rem;
}

.page-title-avatar {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  object-fit: cover;
  border: 1px solid var(--lightgray);
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.18);
}
`

export default (() => PageTitle) satisfies QuartzComponentConstructor
